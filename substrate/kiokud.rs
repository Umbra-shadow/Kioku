// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Guardianity — Kioku v1
// kiokud.rs — Kioku v1 daemon: a line-protocol server over the Cadran substrate.
//
// Owns one `TheBox` (1 TiB sparse vRAM universe + 4 TiB virtual disk) and
// serves newline-delimited JSON over a Unix socket. std only — builds with
// bare rustc, same as the rest of the substrate.
//
// Build:  rustc --edition=2021 -C opt-level=3 kiokud.rs -o kiokud
// Run:    KIOKUD_SOCKET=/tmp/kiokud.sock KIOKUD_DISK=./kioku_box.disk ./kiokud
// Test:   rustc --edition=2021 --test kiokud.rs -o kiokud_tests && ./kiokud_tests
//
// Env:    KIOKUD_SOCKET         socket path        (default /tmp/kiokud.sock)
//         KIOKUD_DISK           virtual disk file  (default ./kioku_box.disk)
//         KIOKUD_CEILING_BYTES  host-safe ceiling  (default 50% of host RAM)
//
// Protocol — one JSON object per line in, one per line out:
//   {"op":"ping"}                                    -> {"ok":true,"pong":true}
//   {"op":"open_space","budget":B}                   -> {"ok":true,"space":N}
//   {"op":"put","space":N,"cells":[{"cell":C,"act":A,"expert":E,"weight":W},..]}
//                                  -> {"ok":true,"written":n,"within_budget":bool}
//   {"op":"get","space":N,"cell":C}                  -> {"ok":true,"found":bool,...}
//   {"op":"scan","space":N,"start":S,"count":K}      -> {"ok":true,"cells":[..]}
//   {"op":"put_blob","space":N,"b64":...}            -> {"ok":true,"block":B,"len":L}
//   {"op":"get_blob","space":N,"block":B,"len":L}    -> {"ok":true,"b64":...}
//   {"op":"check_budget","space":N}  -> {"ok":true,"within":bool,"committed":..,"budget":..}
//   {"op":"stats"}                                   -> {"ok":true, gauges...}
//   {"op":"release_space","space":N}                 -> {"ok":true,"freed":F}
// Any failure: {"ok":false,"error":"..."}.
//
// Addressing discipline lives in the engine: keyword index cells sit at
// `hash64(keyword) & PLANET_CELL_MASK`, so a lookup is one shift+mask jump.
// The daemon just moves cells and blobs; it never searches.

#[allow(dead_code)]
mod cadran_storage;
#[allow(dead_code)]
mod cadran_vram;
#[allow(dead_code)]
mod space;

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex};

use cadran_storage::DiskObjectHandle;
use space::{Capabilities, SpaceId, TheBox};

const MAX_LINE_BYTES: usize = 32 << 20; // one request line
const MAX_CELLS_PER_OP: usize = 1 << 16;
const MAX_SCAN_COUNT: u64 = 1 << 16;
const MAX_BLOB_BYTES: usize = 16 << 20;

// ---------------------------------------------------------------------------
// Minimal JSON — only what the line protocol needs. Unsigned integers are
// kept exact (engram keys are full u64s; f64 would silently round them).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    UInt(u64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Json::UInt(u) => Some(*u),
            Json::Float(f) if *f >= 0.0 && f.fract() == 0.0 && *f <= 2f64.powi(53) => {
                Some(*f as u64)
            }
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Json::UInt(u) => Some(*u as f64),
            Json::Float(f) => Some(*f),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
}

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

pub fn json_parse(input: &str) -> Result<Json, String> {
    let mut p = JsonParser {
        b: input.as_bytes(),
        i: 0,
    };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing bytes at offset {}", p.i));
    }
    Ok(v)
}

impl<'a> JsonParser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at offset {}", c as char, self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(format!("unexpected byte at offset {}", self.i)),
        }
    }

    fn literal(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("bad literal at offset {}", self.i))
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        let mut float = false;
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' => self.i += 1,
                b'-' | b'+' if self.i == start => self.i += 1,
                b'.' | b'e' | b'E' | b'+' | b'-' => {
                    float = true;
                    self.i += 1;
                }
                _ => break,
            }
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?;
        if !float && !s.starts_with('-') {
            if let Ok(u) = s.parse::<u64>() {
                return Ok(Json::UInt(u));
            }
        }
        s.parse::<f64>()
            .map(Json::Float)
            .map_err(|_| format!("bad number '{}' at offset {}", s, start))
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self.peek().ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                self.expect(b'\\')?;
                                self.expect(b'u')?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err("bad surrogate pair".into());
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).ok_or("bad codepoint")?);
                        }
                        _ => return Err(format!("bad escape at offset {}", self.i)),
                    }
                }
                _ => {
                    // Re-decode the UTF-8 sequence starting at c.
                    let len = match c {
                        0x00..=0x7F => 1,
                        0xC0..=0xDF => 2,
                        0xE0..=0xEF => 3,
                        0xF0..=0xF7 => 4,
                        _ => return Err("bad utf-8".into()),
                    };
                    let start = self.i - 1;
                    self.i = start + len;
                    if self.i > self.b.len() {
                        return Err("truncated utf-8".into());
                    }
                    let s =
                        std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?;
                    out.push_str(s);
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("truncated \\u escape".into());
        }
        let s = std::str::from_utf8(&self.b[self.i..self.i + 4]).map_err(|e| e.to_string())?;
        self.i += 4;
        u32::from_str_radix(s, 16).map_err(|_| "bad \\u escape".into())
    }

    fn object(&mut self) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            self.expect(b':')?;
            self.ws();
            let val = self.value()?;
            pairs.push((key, val));
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(pairs));
                }
                _ => return Err(format!("expected ',' or '}}' at offset {}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at offset {}", self.i)),
            }
        }
    }
}

pub fn json_write(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::UInt(u) => out.push_str(&u.to_string()),
        Json::Float(f) => {
            if f.is_finite() {
                // {:?} is shortest-roundtrip for f64 on stable Rust.
                out.push_str(&format!("{:?}", f));
            } else {
                out.push_str("null");
            }
        }
        Json::Str(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Json::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json_write(item, out);
            }
            out.push(']');
        }
        Json::Obj(pairs) => {
            out.push('{');
            for (i, (k, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json_write(&Json::Str(k.clone()), out);
                out.push(':');
                json_write(val, out);
            }
            out.push('}');
        }
    }
}

pub fn json_to_string(v: &Json) -> String {
    let mut s = String::new();
    json_write(v, &mut s);
    s
}

// ---------------------------------------------------------------------------
// Base64 (standard alphabet, padded) — blobs travel as b64 strings.
// ---------------------------------------------------------------------------

const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(B64_ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(B64_ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64_ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64_ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

pub fn b64_decode(s: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Result<u32, String> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("bad base64 byte 0x{:02x}", c)),
        }
    }
    let s = s.trim_end_matches('=').as_bytes();
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    for chunk in s.chunks(4) {
        if chunk.len() == 1 {
            return Err("truncated base64".into());
        }
        let mut n = 0u32;
        for &c in chunk {
            n = (n << 6) | val(c)?;
        }
        n <<= 6 * (4 - chunk.len()) as u32;
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Request handling
// ---------------------------------------------------------------------------

type Fields = Vec<(&'static str, Json)>;

fn respond_ok(mut fields: Fields) -> Json {
    let mut pairs = vec![("ok".to_string(), Json::Bool(true))];
    pairs.extend(fields.drain(..).map(|(k, v)| (k.to_string(), v)));
    Json::Obj(pairs)
}

fn respond_err(msg: String) -> Json {
    Json::Obj(vec![
        ("ok".to_string(), Json::Bool(false)),
        ("error".to_string(), Json::Str(msg)),
    ])
}

fn need_u64(req: &Json, key: &str) -> Result<u64, String> {
    req.get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("missing or invalid '{}'", key))
}

fn need_space(req: &Json) -> Result<SpaceId, String> {
    Ok(SpaceId(need_u64(req, "space")? as u32))
}

pub fn handle_request(the_box: &mut TheBox, req: &Json) -> Json {
    let op = match req.get("op").and_then(|v| v.as_str()) {
        Some(o) => o,
        None => return respond_err("missing 'op'".into()),
    };
    let result = match op {
        "ping" => Ok(vec![("pong", Json::Bool(true))]),
        "open_space" => op_open_space(the_box, req),
        "put" => op_put(the_box, req),
        "get" => op_get(the_box, req),
        "scan" => op_scan(the_box, req),
        "put_blob" => op_put_blob(the_box, req),
        "get_blob" => op_get_blob(the_box, req),
        "check_budget" => op_check_budget(the_box, req),
        "stats" => op_stats(the_box),
        "release_space" => op_release_space(the_box, req),
        other => Err(format!("unknown op '{}'", other)),
    };
    match result {
        Ok(fields) => respond_ok(fields),
        Err(e) => respond_err(e),
    }
}

fn op_open_space(the_box: &mut TheBox, req: &Json) -> Result<Fields, String> {
    let budget = need_u64(req, "budget")?;
    let id = the_box
        .open_space(budget, Capabilities::default())
        .map_err(|e| e.to_string())?;
    Ok(vec![("space", Json::UInt(id.0 as u64))])
}

fn op_put(the_box: &mut TheBox, req: &Json) -> Result<Fields, String> {
    let id = need_space(req)?;
    let cells = req
        .get("cells")
        .and_then(|v| v.as_arr())
        .ok_or("missing or invalid 'cells'")?;
    if cells.len() > MAX_CELLS_PER_OP {
        return Err(format!("too many cells (max {})", MAX_CELLS_PER_OP));
    }
    let mut parsed = Vec::with_capacity(cells.len());
    for c in cells {
        let cell = need_u64(c, "cell")?;
        let act = c
            .get("act")
            .and_then(|v| v.as_f64())
            .ok_or("missing or invalid 'act'")? as f32;
        let expert = need_u64(c, "expert")?;
        if expert > u32::MAX as u64 {
            return Err("'expert' exceeds u32".into());
        }
        let weight = need_u64(c, "weight")?;
        parsed.push((cell, act, expert as u32, weight));
    }
    let mut h = the_box.space(id).map_err(|e| e.to_string())?;
    for (cell, act, expert, weight) in &parsed {
        h.write_cell(*cell, *act, *expert, *weight);
    }
    let within = h.check_budget().is_ok();
    Ok(vec![
        ("written", Json::UInt(parsed.len() as u64)),
        ("within_budget", Json::Bool(within)),
    ])
}

fn op_get(the_box: &mut TheBox, req: &Json) -> Result<Fields, String> {
    let id = need_space(req)?;
    let cell = need_u64(req, "cell")?;
    let h = the_box.space(id).map_err(|e| e.to_string())?;
    Ok(match h.peek_cell_full(cell) {
        Some((act, expert, weight)) => vec![
            ("found", Json::Bool(true)),
            ("act", Json::Float(act as f64)),
            ("expert", Json::UInt(expert as u64)),
            ("weight", Json::UInt(weight)),
        ],
        None => vec![("found", Json::Bool(false))],
    })
}

/// Committed, non-zero cells in [start, start+count). All-zero cells are
/// indistinguishable from never-written and are skipped.
fn op_scan(the_box: &mut TheBox, req: &Json) -> Result<Fields, String> {
    let id = need_space(req)?;
    let start = need_u64(req, "start")?;
    let count = need_u64(req, "count")?.min(MAX_SCAN_COUNT);
    let h = the_box.space(id).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for cell in start..start.saturating_add(count) {
        if let Some((act, expert, weight)) = h.peek_cell_full(cell) {
            if act != 0.0 || expert != 0 || weight != 0 {
                out.push(Json::Obj(vec![
                    ("cell".to_string(), Json::UInt(cell)),
                    ("act".to_string(), Json::Float(act as f64)),
                    ("expert".to_string(), Json::UInt(expert as u64)),
                    ("weight".to_string(), Json::UInt(weight)),
                ]));
            }
        }
    }
    Ok(vec![("cells", Json::Arr(out))])
}

fn op_put_blob(the_box: &mut TheBox, req: &Json) -> Result<Fields, String> {
    let id = need_space(req)?;
    let b64 = req
        .get("b64")
        .and_then(|v| v.as_str())
        .ok_or("missing or invalid 'b64'")?;
    let payload = b64_decode(b64)?;
    if payload.len() > MAX_BLOB_BYTES {
        return Err(format!("blob too large (max {} bytes)", MAX_BLOB_BYTES));
    }
    let mut h = the_box.space(id).map_err(|e| e.to_string())?;
    let handle = h.put_paper(&payload).map_err(|e| e.to_string())?;
    Ok(vec![
        ("block", Json::UInt(handle.first_block)),
        ("len", Json::UInt(handle.len)),
    ])
}

fn op_get_blob(the_box: &mut TheBox, req: &Json) -> Result<Fields, String> {
    let id = need_space(req)?;
    let block = need_u64(req, "block")?;
    let len = need_u64(req, "len")?;
    let handle = DiskObjectHandle {
        planet_id: id.0,
        first_block: block,
        len,
    };
    let mut h = the_box.space(id).map_err(|e| e.to_string())?;
    let payload = h.get_paper(handle).map_err(|e| e.to_string())?;
    Ok(vec![("b64", Json::Str(b64_encode(&payload)))])
}

fn op_check_budget(the_box: &mut TheBox, req: &Json) -> Result<Fields, String> {
    let id = need_space(req)?;
    let budget = the_box.space_budget_bytes(id).map_err(|e| e.to_string())?;
    let h = the_box.space(id).map_err(|e| e.to_string())?;
    let committed = h.committed_bytes();
    Ok(vec![
        ("within", Json::Bool(committed <= budget)),
        ("committed", Json::UInt(committed)),
        ("budget", Json::UInt(budget)),
    ])
}

fn op_stats(the_box: &mut TheBox) -> Result<Fields, String> {
    let ids = the_box.open_space_ids();
    let mut spaces = Vec::with_capacity(ids.len());
    for id in ids {
        let budget = the_box.space_budget_bytes(id).map_err(|e| e.to_string())?;
        let committed = the_box.space(id).map_err(|e| e.to_string())?.committed_bytes();
        spaces.push(Json::Obj(vec![
            ("space".to_string(), Json::UInt(id.0 as u64)),
            ("budget".to_string(), Json::UInt(budget)),
            ("committed".to_string(), Json::UInt(committed)),
        ]));
    }
    let disk_committed = the_box.disk_committed_bytes().map_err(|e| e.to_string())?;
    Ok(vec![
        ("backend", Json::Str("kiokud".into())),
        ("vram_committed", Json::UInt(the_box.committed_bytes())),
        ("vram_virtual", Json::UInt(the_box.vram_virtual_bytes())),
        ("disk_committed", Json::UInt(disk_committed)),
        ("disk_virtual", Json::UInt(the_box.disk_virtual_bytes())),
        ("reserved", Json::UInt(the_box.reserved_bytes())),
        ("ceiling", Json::UInt(the_box.ceiling_bytes())),
        ("open_spaces", Json::UInt(the_box.open_spaces() as u64)),
        ("spaces", Json::Arr(spaces)),
    ])
}

fn op_release_space(the_box: &mut TheBox, req: &Json) -> Result<Fields, String> {
    let id = need_space(req)?;
    let freed = the_box.release_space(id).map_err(|e| e.to_string())?;
    Ok(vec![("freed", Json::UInt(freed))])
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

fn handle_line(the_box: &Arc<Mutex<TheBox>>, line: &str) -> String {
    let response = match json_parse(line) {
        Ok(req) => {
            let mut guard = the_box.lock().unwrap_or_else(|p| p.into_inner());
            handle_request(&mut guard, &req)
        }
        Err(e) => respond_err(format!("bad json: {}", e)),
    };
    json_to_string(&response)
}

fn handle_client(stream: UnixStream, the_box: Arc<Mutex<TheBox>>) {
    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(_) => return,
    };
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => return, // client gone
            Ok(_) => {}
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let out = if line.len() > MAX_LINE_BYTES {
            json_to_string(&respond_err("request line too long".into()))
        } else {
            handle_line(&the_box, trimmed)
        };
        if writer
            .write_all(out.as_bytes())
            .and_then(|_| writer.write_all(b"\n"))
            .is_err()
        {
            return;
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn main() {
    let socket_path = env_or("KIOKUD_SOCKET", "/tmp/kiokud.sock");
    let disk_path = env_or("KIOKUD_DISK", "kioku_box.disk");

    let the_box = match std::env::var("KIOKUD_CEILING_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        Some(ceiling) => TheBox::new(&disk_path, ceiling),
        None => TheBox::with_host_fraction(&disk_path, 0.5),
    };
    let the_box = match the_box {
        Ok(b) => b,
        Err(e) => {
            eprintln!("kiokud: cannot open the box at '{}': {}", disk_path, e);
            std::process::exit(1);
        }
    };

    let _ = std::fs::remove_file(&socket_path);
    let listener = match UnixListener::bind(&socket_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("kiokud: cannot bind '{}': {}", socket_path, e);
            std::process::exit(1);
        }
    };

    eprintln!(
        "kiokud: up · socket={} · disk={} · ceiling={} B · vram 1 TiB virtual · disk 4 TiB virtual",
        socket_path,
        disk_path,
        the_box.ceiling_bytes()
    );

    let shared = Arc::new(Mutex::new(the_box));
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let b = Arc::clone(&shared);
                std::thread::spawn(move || handle_client(s, b));
            }
            Err(e) => eprintln!("kiokud: accept error: {}", e),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod kiokud_tests {
    use super::*;

    fn temp_disk(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("kiokud_test_{}_{}", std::process::id(), name));
        p
    }

    fn test_box(name: &str) -> (TheBox, std::path::PathBuf) {
        let path = temp_disk(name);
        let _ = std::fs::remove_file(&path);
        (TheBox::new(&path, 256 << 20).unwrap(), path)
    }

    fn call(the_box: &mut TheBox, req: &str) -> Json {
        handle_request(the_box, &json_parse(req).unwrap())
    }

    fn ok(v: &Json) -> bool {
        v.get("ok") == Some(&Json::Bool(true))
    }

    #[test]
    fn json_roundtrip() {
        let src = r#"{"op":"put","space":3,"cells":[{"cell":18446744073709551615,"act":-1.5,"expert":7,"weight":42}],"note":"a\nb\"c\\d","none":null,"flag":true}"#;
        let v = json_parse(src).unwrap();
        let re = json_parse(&json_to_string(&v)).unwrap();
        assert_eq!(v, re);
        // u64 keys survive exactly — no f64 rounding.
        let cells = v.get("cells").unwrap().as_arr().unwrap();
        assert_eq!(cells[0].get("cell").unwrap().as_u64(), Some(u64::MAX));
    }

    #[test]
    fn json_unicode_escapes() {
        let v = json_parse(r#"{"s":"é😀 記憶"}"#).unwrap();
        assert_eq!(v.get("s").unwrap().as_str(), Some("é😀 記憶"));
        let re = json_parse(&json_to_string(&v)).unwrap();
        assert_eq!(v, re);
    }

    #[test]
    fn json_rejects_garbage() {
        assert!(json_parse("{").is_err());
        assert!(json_parse(r#"{"a":}"#).is_err());
        assert!(json_parse("[1,2,]").is_err());
        assert!(json_parse("{} extra").is_err());
    }

    #[test]
    fn base64_vectors_and_roundtrip() {
        assert_eq!(b64_encode(b""), "");
        assert_eq!(b64_encode(b"f"), "Zg==");
        assert_eq!(b64_encode(b"fo"), "Zm8=");
        assert_eq!(b64_encode(b"foo"), "Zm9v");
        assert_eq!(b64_decode("Zm9vYmFy").unwrap(), b"foobar");
        let data: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        assert_eq!(b64_decode(&b64_encode(&data)).unwrap(), data);
        assert!(b64_decode("@@@@").is_err());
    }

    #[test]
    fn full_flow_open_put_get_scan_release() {
        let (mut b, path) = test_box("flow");
        let r = call(&mut b, r#"{"op":"open_space","budget":16777216}"#);
        assert!(ok(&r));
        let space = r.get("space").unwrap().as_u64().unwrap();

        let put = format!(
            r#"{{"op":"put","space":{},"cells":[{{"cell":4242,"act":0.75,"expert":3,"weight":999}},{{"cell":4243,"act":1.0,"expert":0,"weight":1}}]}}"#,
            space
        );
        let r = call(&mut b, &put);
        assert!(ok(&r));
        assert_eq!(r.get("written").unwrap().as_u64(), Some(2));
        assert_eq!(r.get("within_budget"), Some(&Json::Bool(true)));

        let r = call(&mut b, &format!(r#"{{"op":"get","space":{},"cell":4242}}"#, space));
        assert!(ok(&r));
        assert_eq!(r.get("found"), Some(&Json::Bool(true)));
        assert_eq!(r.get("act").unwrap().as_f64(), Some(0.75));
        assert_eq!(r.get("expert").unwrap().as_u64(), Some(3));
        assert_eq!(r.get("weight").unwrap().as_u64(), Some(999));

        // Untouched cell in an untouched segment: found=false.
        let r = call(
            &mut b,
            &format!(r#"{{"op":"get","space":{},"cell":4000000}}"#, space),
        );
        assert_eq!(r.get("found"), Some(&Json::Bool(false)));

        let r = call(
            &mut b,
            &format!(r#"{{"op":"scan","space":{},"start":4200,"count":100}}"#, space),
        );
        assert!(ok(&r));
        assert_eq!(r.get("cells").unwrap().as_arr().unwrap().len(), 2);

        let r = call(&mut b, &format!(r#"{{"op":"release_space","space":{}}}"#, space));
        assert!(ok(&r));
        assert!(r.get("freed").unwrap().as_u64().unwrap() > 0);

        // Space is closed now.
        let r = call(&mut b, &format!(r#"{{"op":"get","space":{},"cell":4242}}"#, space));
        assert!(!ok(&r));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn blob_roundtrip_and_cross_space_refused() {
        let (mut b, path) = test_box("blob");
        let s1 = call(&mut b, r#"{"op":"open_space","budget":8388608}"#)
            .get("space")
            .unwrap()
            .as_u64()
            .unwrap();
        let s2 = call(&mut b, r#"{"op":"open_space","budget":8388608}"#)
            .get("space")
            .unwrap()
            .as_u64()
            .unwrap();

        let payload = b"an engram, serialized";
        let put = format!(
            r#"{{"op":"put_blob","space":{},"b64":"{}"}}"#,
            s1,
            b64_encode(payload)
        );
        let r = call(&mut b, &put);
        assert!(ok(&r));
        let block = r.get("block").unwrap().as_u64().unwrap();
        let len = r.get("len").unwrap().as_u64().unwrap();
        assert_eq!(len, payload.len() as u64);

        let r = call(
            &mut b,
            &format!(r#"{{"op":"get_blob","space":{},"block":{},"len":{}}}"#, s1, block, len),
        );
        assert!(ok(&r));
        assert_eq!(
            b64_decode(r.get("b64").unwrap().as_str().unwrap()).unwrap(),
            payload
        );

        // Another space cannot read it: nothing lives at that block in s2.
        let r = call(
            &mut b,
            &format!(r#"{{"op":"get_blob","space":{},"block":{},"len":{}}}"#, s2, block, len),
        );
        assert!(!ok(&r));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn spaces_are_isolated_through_the_protocol() {
        let (mut b, path) = test_box("iso");
        let s1 = call(&mut b, r#"{"op":"open_space","budget":8388608}"#)
            .get("space")
            .unwrap()
            .as_u64()
            .unwrap();
        let s2 = call(&mut b, r#"{"op":"open_space","budget":8388608}"#)
            .get("space")
            .unwrap()
            .as_u64()
            .unwrap();
        call(
            &mut b,
            &format!(r#"{{"op":"put","space":{},"cells":[{{"cell":7,"act":1.5,"expert":1,"weight":1}}]}}"#, s1),
        );
        let r = call(&mut b, &format!(r#"{{"op":"get","space":{},"cell":7}}"#, s2));
        assert_eq!(r.get("found"), Some(&Json::Bool(false)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn budget_and_stats() {
        let (mut b, path) = test_box("stats");
        let s = call(&mut b, r#"{"op":"open_space","budget":1048576}"#)
            .get("space")
            .unwrap()
            .as_u64()
            .unwrap();
        call(
            &mut b,
            &format!(r#"{{"op":"put","space":{},"cells":[{{"cell":0,"act":1.0,"expert":0,"weight":1}}]}}"#, s),
        );
        let r = call(&mut b, &format!(r#"{{"op":"check_budget","space":{}}}"#, s));
        assert!(ok(&r));
        assert_eq!(r.get("within"), Some(&Json::Bool(true)));

        let r = call(&mut b, r#"{"op":"stats"}"#);
        assert!(ok(&r));
        assert_eq!(r.get("vram_virtual").unwrap().as_u64(), Some(1u64 << 40));
        assert_eq!(r.get("disk_virtual").unwrap().as_u64(), Some(1u64 << 42));
        assert_eq!(r.get("open_spaces").unwrap().as_u64(), Some(1));
        assert!(r.get("vram_committed").unwrap().as_u64().unwrap() > 0);
        let spaces = r.get("spaces").unwrap().as_arr().unwrap();
        assert_eq!(spaces.len(), 1);
        assert_eq!(spaces[0].get("space").unwrap().as_u64(), Some(s));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn protocol_errors_are_errors_not_panics() {
        let (mut b, path) = test_box("err");
        assert!(!ok(&call(&mut b, r#"{"op":"warp"}"#)));
        assert!(!ok(&call(&mut b, r#"{"nope":1}"#)));
        assert!(!ok(&call(&mut b, r#"{"op":"put","space":99,"cells":[]}"#)));
        assert!(!ok(&call(&mut b, r#"{"op":"open_space"}"#)));
        assert!(!ok(&call(
            &mut b,
            r#"{"op":"put_blob","space":1,"b64":"!!!"}"#
        )));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn serves_over_a_real_unix_socket() {
        let disk = temp_disk("sock_disk");
        let _ = std::fs::remove_file(&disk);
        let mut sock = std::env::temp_dir();
        sock.push(format!("kiokud_test_{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);

        let the_box = Arc::new(Mutex::new(TheBox::new(&disk, 64 << 20).unwrap()));
        let listener = UnixListener::bind(&sock).unwrap();
        let served = Arc::clone(&the_box);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let b = Arc::clone(&served);
                std::thread::spawn(move || handle_client(stream, b));
            }
        });

        let stream = UnixStream::connect(&sock).unwrap();
        let mut writer = stream.try_clone().unwrap();
        let mut reader = BufReader::new(stream);
        let mut send = |req: &str| -> Json {
            writer.write_all(req.as_bytes()).unwrap();
            writer.write_all(b"\n").unwrap();
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            json_parse(line.trim()).unwrap()
        };

        assert_eq!(send(r#"{"op":"ping"}"#).get("pong"), Some(&Json::Bool(true)));
        let s = send(r#"{"op":"open_space","budget":4194304}"#)
            .get("space")
            .unwrap()
            .as_u64()
            .unwrap();
        let r = send(&format!(
            r#"{{"op":"put","space":{},"cells":[{{"cell":1,"act":2.5,"expert":9,"weight":77}}]}}"#,
            s
        ));
        assert!(ok(&r));
        let r = send(&format!(r#"{{"op":"get","space":{},"cell":1}}"#, s));
        assert_eq!(r.get("act").unwrap().as_f64(), Some(2.5));
        assert_eq!(r.get("weight").unwrap().as_u64(), Some(77));

        let _ = std::fs::remove_file(&sock);
        let _ = std::fs::remove_file(&disk);
    }
}
