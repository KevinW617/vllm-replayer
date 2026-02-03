#!/usr/bin/env python3
"""
trace_replay.py

Replay a trajectory JSON that contains timestamps of vLLM requests and tool calls.

Usage:
  python trace_replay.py /path/to/trajectory.json [--vllm-url URL] [--threshold SECONDS] [--dry-run]

Behavior:
 - For events that correspond to vLLM calls (matched to files in ../llm_completions/<basename>), the script will extract the original prompt/messages and either print them (--dry-run) or POST them to the provided vLLM URL.
 - For other events (treated as tool calls), the script will busy-wait (CPU spin) for the duration until the next event to emulate CPU-bound tool execution.

Notes:
 - To actually send to a vLLM, provide a vllm-compatible HTTP endpoint via --vllm-url (e.g. http://localhost:8000/v1/chat/completions). The script will POST JSON with a 'messages' field when available.
 - If --dry-run is set or no --vllm-url is provided, the script will only print prompts and actions.

This tool is conservative about matching llm files. It uses file metadata and common JSON fields like 'created', 'timestamp', or file mtime.
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from urllib import request, error

MAX_DURATION = sys.float_info.max / 1e3

# Hard-coded vLLM configuration (per user request)
VLLM_CFG = {
    'model': 'mistralai/Devstral-Small-2507',
    'base_url': 'http://localhost:8000/v1',
    'api_key': 'abc-123',
    'seed': 42,
    'temperature': 0.0,
    'top_p': 1.0,
    'top_k': -1,
    'stop': ['</function'],
    'max_tokens': 4096,
    'force_expected': True,
    'include_stop_str_in_output': True,
}


def parse_iso_to_epoch(s):
    # support ISO with or without timezone
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        # fallback: try removing Z then parse
        if s.endswith("Z"):
            s2 = s[:-1]
            dt = datetime.fromisoformat(s2)
        else:
            raise
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def find_llm_dir_for_trace(trace_path):
    # expected: trace is .../trajectories/<name>.json and llm files are ../llm_completions/<name>/*
    base = os.path.splitext(os.path.basename(trace_path))[0]
    parent = os.path.dirname(trace_path)
    # try sibling llm_completions/<base>
    candidate = os.path.join(parent, '..', 'llm_completions', base)
    candidate = os.path.normpath(candidate)
    if os.path.isdir(candidate):
        return candidate
    # try ../llm_completions/<base>/ (two levels up)
    candidate2 = os.path.join(parent, '..', '..', 'llm_completions', base)
    candidate2 = os.path.normpath(candidate2)
    if os.path.isdir(candidate2):
        return candidate2
    # also try sibling folder in parent of parent
    return None


def load_llm_files(llm_dir):
    entries = []
    if not llm_dir:
        return entries
    for fname in os.listdir(llm_dir):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(llm_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            # skip broken files
            data = None
        # determine timestamp (epoch seconds)
        ts = None
        if isinstance(data, dict):
            # common fields
            for key in ('created', 'timestamp'):
                if key in data:
                    v = data[key]
                    try:
                        ts = float(v)
                        # if ts looks like epoch > 1e12 maybe ms
                        if ts > 1e12:
                            ts = ts / 1000.0
                        break
                    except Exception:
                        pass
            # nested 'response.created' or similar
            if ts is None:
                if 'response' in data and isinstance(data['response'], dict):
                    v = data['response'].get('created')
                    if v is not None:
                        try:
                            ts = float(v)
                            if ts > 1e12:
                                ts = ts / 1000.0
                        except Exception:
                            ts = None
        if ts is None:
            # fallback to file mtime
            try:
                ts = os.path.getmtime(path)
            except Exception:
                ts = None
        entries.append({'path': path, 'data': data, 'ts': ts})
    # sort by timestamp
    entries.sort(key=lambda e: e['ts'] or 0)
    return entries


def extract_messages_from_llm_file(entry):
    data = entry['data']
    if not isinstance(data, dict):
        return None
    # If this looks like OpenAI chat format
    if 'messages' in data and isinstance(data['messages'], list):
        return data['messages']
    # If the file contains 'messages' inside another field (e.g., wrapper)
    for key in ('request', 'input'):
        if key in data and isinstance(data[key], dict) and 'messages' in data[key]:
            return data[key]['messages']
    # If there's a single 'prompt' or 'messages' style
    if 'prompt' in data:
        return [{'role': 'user', 'content': data['prompt']}]
    # fallback: pretty-print the file as a single user message
    try:
        s = json.dumps(data)
    except Exception:
        s = str(data)
    return [{'role': 'user', 'content': s}]


def post_to_vllm(messages,
                 session_id=None,
                 next_arrival_delta=None,
                 expected_resp=None,
                 timeout=1200):
    """Post messages to the hard-coded vLLM endpoint.

    The function uses VLLM_CFG['base_url'] and VLLM_CFG['api_key'] for the request.

    Args:
        messages: List of message dictionaries for the chat completion
        session_id: Optional session identifier to group related requests
        next_arrival_delta: Optional predicted time until next request in this session
        timeout: Request timeout in seconds
    """
    base = VLLM_CFG.get('base_url')
    if not base:
        return None, 'No base_url configured', None
    # use chat completions endpoint by default
    url = base.rstrip('/') + '/chat/completions'
    payload_body = {
        'model': VLLM_CFG.get('model'),
        'messages': messages,
        'seed': VLLM_CFG.get('seed'),
        'temperature': VLLM_CFG.get('temperature'),
        'top_p': VLLM_CFG.get('top_p'),
        'top_k': VLLM_CFG.get('top_k'),
        'stop': VLLM_CFG.get('stop'),
        'max_tokens': VLLM_CFG.get('max_tokens'),
    }

    # Add a unique request_id for vLLM tracking (mirrors OpenHands LLM wrapper behavior)
    # Use session_id as the instance identifier when available.
    instance_id = session_id
    if instance_id:
        request_id = f'{instance_id}_{uuid.uuid4().hex[:8]}'
    else:
        request_id = uuid.uuid4().hex
    # NOTE: OpenHands sets this under litellm's `extra_body`, and litellm expands it into the
    # top-level OpenAI-compatible JSON body. This script POSTs directly (no litellm), so we
    # explicitly set the top-level field for vLLM, and also keep extra_body for parity.
    payload_body['request_id'] = request_id
    payload_body['extra_body'] = {'request_id': request_id}

    if session_id is not None or next_arrival_delta is not None or expected_resp is not None:
        vllm_xargs = {}
        if session_id is not None:
            vllm_xargs['session_id'] = session_id
        if next_arrival_delta is not None:
            vllm_xargs['next_arrival_delta'] = next_arrival_delta
        if expected_resp is not None:
            vllm_xargs['expected_resp'] = expected_resp
        vllm_xargs['force_expected'] = True
        payload_body['vllm_xargs'] = vllm_xargs

    payload = json.dumps(payload_body).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    api_key = VLLM_CFG.get('api_key')
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    req = request.Request(url, data=payload, headers=headers)
    try:
        t0 = time.time()
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode('utf-8', errors='replace')
            dt = time.time() - t0
            return resp.getcode(), body, dt
    except error.HTTPError as e:
        return e.code, e.read().decode('utf-8', errors='replace'), None
    except Exception as e:
        return None, str(e), None


def extract_response_text(body):
    """Try to extract a readable text response from a vLLM/LLM server body.

    Returns a string.
    """
    if not body:
        return ''
    try:
        data = json.loads(body)
    except Exception:
        return str(body)
    # OpenAI-like: choices -> message -> content
    if isinstance(data, dict):
        if 'choices' in data and isinstance(data['choices'], list) and data['choices']:
            parts = []
            for c in data['choices']:
                # support 'message' with 'content'
                if isinstance(c, dict):
                    if 'message' in c and isinstance(c['message'], dict) and 'content' in c['message']:
                        parts.append(str(c['message']['content']))
                        continue
                    # older format: 'text'
                    if 'text' in c:
                        parts.append(str(c['text']))
                        continue
                    # delta streaming
                    if 'delta' in c and isinstance(c['delta'], dict) and 'content' in c['delta']:
                        parts.append(str(c['delta']['content']))
                        continue
            if parts:
                return '\n'.join(parts)
        # vLLM might return 'output' or 'response' keys
        for key in ('output', 'response', 'result'):
            if key in data:
                v = data[key]
                try:
                    return json.dumps(v) if not isinstance(v, str) else v
                except Exception:
                    return str(v)
    # fallback to stringified JSON
    try:
        return json.dumps(data)
    except Exception:
        return str(data)


def get_next_llm_after(ts, llm_entries):
    if ts is None:
        return None
    for entry in llm_entries:
        if entry.get('ts') is None:
            continue
        if entry['ts'] > ts:
            return entry
    return None


def simulate_tool_call(duration):
    time.sleep(duration)
    return None


def main():
    p = argparse.ArgumentParser(description='Replay a trajectory of vLLM and tool events')
    p.add_argument('trajectory', help='Path to trajectory JSON file')
    # vLLM is hard-coded via VLLM_CFG
    p.add_argument('--threshold', type=float, default=3.0, help='max seconds to match a trajectory event to an llm file')
    p.add_argument('--dry-run', action='store_true', help='Do not send requests to vLLM; only print actions')
    p.add_argument('--verbose', '-v', action='store_true')
    p.add_argument('--out', dest='out_path', help='Path to write replayed trajectory JSON (default: same directory as input, <basename>.replayed.json)', default=None)
    args = p.parse_args()

    traj_path = args.trajectory
    if not os.path.isfile(traj_path):
        print('trajectory file not found:', traj_path, file=sys.stderr)
        sys.exit(2)

    with open(traj_path, 'r', encoding='utf-8') as f:
        traj = json.load(f)

    # normalize events into list of dict with timestamp as epoch
    events = []
    for ev in traj:
        ts = None
        if 'timestamp' in ev:
            try:
                ts = parse_iso_to_epoch(ev['timestamp'])
            except Exception:
                # try numeric
                try:
                    ts = float(ev['timestamp'])
                except Exception:
                    ts = None
        # fallback: if event has numeric 'time' or 'created'
        for key in ('time', 'created'):
            if ts is None and key in ev:
                try:
                    ts = float(ev[key])
                except Exception:
                    pass
        events.append({'raw': ev, 'ts': ts})

    # sort by ts
    events.sort(key=lambda e: e['ts'] or 0)

    llm_dir = find_llm_dir_for_trace(traj_path)
    if args.verbose:
        print('Looking for llm files in', llm_dir)
    llm_entries = load_llm_files(llm_dir)

    # keep a flag whether an llm entry was used
    for e in llm_entries:
        e['_used'] = False

    # Derive session_id from trajectory filename
    session_id = os.path.splitext(os.path.basename(traj_path))[0]

    # Collect all inference event indices and their timestamps for computing next_arrival_delta
    inference_indices = []
    for i, ev in enumerate(events):
        ts = ev.get('ts')
        raw = ev.get('raw')
        if ts is not None and isinstance(raw, dict) and 'action' in raw:
            inference_indices.append(i)

    # iterate events
    n = len(events)
    out_events = []
    out_id = 0
    for i, ev in enumerate(events):
        ts = ev['ts']
        raw = ev['raw']
        next_ts = None
        if i + 1 < n:
            next_ts = events[i + 1]['ts']
        # Decide whether this event is an inference (vLLM) or a runtime/function call
        # Use explicit fields on the event to disambiguate (preferred over timestamp-only matching):
        # if 'action' in event -> inference
        # elif 'observation' in event and 'cause' in event -> function call / runtime action
        is_inference = False
        is_function_call = False
        if isinstance(raw, dict):
            if 'action' in raw:
                is_inference = True
            elif 'observation' in raw and 'cause' in raw:
                is_function_call = True

        # attempt to match nearest llm entry when this looks like an inference
        matched = None
        best_dt = None
        if is_inference and ts is not None:
            for entry in llm_entries:
                if entry['ts'] is None:
                    continue
                dt = abs(entry['ts'] - ts)
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    matched = entry
            if best_dt is not None and best_dt > args.threshold:
                matched = None

        # If event was not clearly inference/function-call, fall back to previous timestamp-based matching
        if not is_inference and not is_function_call and ts is not None:
            for entry in llm_entries:
                if entry['ts'] is None:
                    continue
                dt = abs(entry['ts'] - ts)
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    matched = entry
            if best_dt is not None and best_dt > args.threshold:
                matched = None

        # If this event is inference and we have a matched llm file, treat it as vLLM call
        if is_inference and matched is not None and not matched.get('_used', False):
            matched['_used'] = True
            messages = extract_messages_from_llm_file(matched)

            # Compute next_arrival_delta: tool call duration (time from current inference end to next inference start)
            # This represents the time spent executing tool calls between inferences
            next_arrival_delta = MAX_DURATION
            try:
                current_idx_in_inferences = inference_indices.index(i)
                if current_idx_in_inferences + 1 < len(inference_indices):
                    next_inference_idx = inference_indices[current_idx_in_inferences + 1]
                    next_inference_ts = events[next_inference_idx].get('ts')
                    if next_inference_ts is not None and ts is not None:
                        # Tool call duration is the time from current inference completion to next inference start
                        # Since event timestamps are completion times, the last tool event before next inference
                        # marks when tool calls completed
                        last_tool_ts = None
                        for k in range(i + 1, next_inference_idx):
                            k_ts = events[k].get('ts')
                            k_raw = events[k].get('raw')
                            if k_ts is not None and isinstance(k_raw, dict):
                                # Tool events don't have 'action' field
                                if not ('action' in k_raw):
                                    last_tool_ts = k_ts

                        # next_arrival_delta is the tool call duration
                        if last_tool_ts is not None:
                            # Time from current inference end to last tool call end
                            next_arrival_delta = float(last_tool_ts - ts)
                        else:
                            # No tool calls between inferences, so delta is time to next inference
                            next_arrival_delta = float(next_inference_ts - ts)
            except (ValueError, IndexError):
                pass

            print(f"[{i}] vLLM call START at time.time() {time.time()} {datetime.now(timezone.utc).isoformat()}")
            expected_resp = get_next_llm_after(matched.get('ts'), llm_entries)
            if expected_resp is not None:
                expected_msgs = extract_messages_from_llm_file(expected_resp)
                expected_text = ''
                if expected_msgs:
                    m = expected_msgs[-2]
                    if isinstance(m, dict):
                        expected_text = str(m.get('content')
                                            or m.get('text') or '')
                    else:
                        expected_text = str(m)
            if args.dry_run:
                # print('--- prompt/messages ---')
                # try:
                #     print(json.dumps(messages, indent=2, ensure_ascii=False))
                # except Exception:
                #     print(messages)
                # print('--- end prompt ---')
                print(f'--- session_id: {session_id}, next_arrival_delta: {next_arrival_delta} ---')
            else:
                code, body, elapsed = post_to_vllm(messages,
                                                  session_id=session_id,
                                                  next_arrival_delta=next_arrival_delta,
                                                  expected_resp=expected_text)
                print(f"[{i}] vLLM call END at time.time() {time.time()} {datetime.now(timezone.utc).isoformat()} for {elapsed:.3f}s")
                print('POST', VLLM_CFG.get('base_url') + '/chat/completions', 'status=', code, 'elapsed=', elapsed)
                if args.verbose:
                    print('response (raw):', body[:2000])
                # extract readable response text and compare to next prompt
                resp_text = extract_response_text(body)
                # try to parse response JSON to extract id and usage if available
                response_id = None
                usage = None
                try:
                    parsed = json.loads(body) if body else {}
                    if isinstance(parsed, dict):
                        response_id = parsed.get('id')
                        usage = parsed.get('usage')
                except Exception:
                    parsed = None

                next_llm = get_next_llm_after(matched.get('ts'), llm_entries)
                if next_llm is not None:
                    next_msgs = extract_messages_from_llm_file(next_llm)
                    # flatten next prompt content for comparison
                    next_text = ''
                    if next_msgs:
                        m = next_msgs[-2]
                        if isinstance(m, dict):
                            next_text = str(m.get('content') or m.get('text') or '').strip()
                        else:
                            next_text = str(m).strip()
                    if next_text:
                        # normalize both texts: strip whitespace and remove a trailing closing function tag if present
                        resp_norm = resp_text.strip()
                        next_norm = next_text.strip()
                        closing_tag = '</function>'
                        if resp_norm.endswith(closing_tag):
                            resp_norm = resp_norm[:-len(closing_tag)].strip()
                        if next_norm.endswith(closing_tag):
                            next_norm = next_norm[:-len(closing_tag)].strip()
                        if resp_norm != next_norm:
                            print('WARNING: vLLM response does not match next prompt (they differ)')
                            print('--------------response_text------------\n', resp_text)
                            print('--------------next_prompt_text--------------\n', next_text)
                            print('--------------expected_resp--------------\n', expected_resp)

                # record output event using the actual simulated timestamp
                sim_ts = datetime.now(timezone.utc).isoformat()
                try:
                    model_response = json.loads(body) if body else None
                except Exception:
                    model_response = body
                out_ev = {
                    'id': out_id,
                    'timestamp': sim_ts,
                    'source': 'agent',
                    'llm_file': os.path.basename(matched.get('path')) if matched.get('path') else None,
                    'model_response': model_response,
                    'response_text': resp_text,
                    'response_elapsed': elapsed,
                }
                out_events.append(out_ev)
                out_id += 1
            # optionally, sleep a short time to mimic pacing
            time.sleep(0.01)
        else:
            # treat as a tool call (duration is event_ts - previous_event_ts; event timestamps are end times)
            # compute previous event timestamp (use 0 for the first event)
            if ts is None:
                if args.verbose:
                    print(f"[{i}] Skipping tool-like event (no timestamp): {raw.get('action') or raw.get('message')}")
                continue
            if i == 0:
                prev_ts = ts
            else:
                prev_ts = events[i - 1].get('ts') or ts

            duration = max(0.0, float(ts) - float(prev_ts))
            if duration <= 0:
                if args.verbose:
                    print(f"[{i}] Non-positive duration ({duration}); skipping")
                continue

            print(f"[{i}] Tool call simulation START at time.time() {time.time()} {datetime.now(timezone.utc).isoformat()} for {duration:.3f}s (calculated from previous event)")
            start = time.time()
            simulate_tool_call(duration)
            took = time.time() - start
            print(f"[{i}] Tool call simulation END at time.time() {time.time()} {datetime.now(timezone.utc).isoformat()} for {duration:.3f}s (requested {duration:.3f}s, actual {took:.3f}s)")
            # record tool event
            # use simulated timestamp for recorded event
            sim_ts = datetime.now(timezone.utc).isoformat()
            out_ev = {
                'id': out_id,
                'timestamp': sim_ts,
                'source': 'tool',
                'duration_requested': duration,
                'duration_actual': took,
            }
            out_events.append(out_ev)
            out_id += 1

    # After processing, write out replay trajectory if not dry-run
    if not args.dry_run:
        # Determine output path: use provided args.out_path if given, otherwise default next to input
        if args.out_path:
            out_path = args.out_path
            # if out_path is a directory, place file inside with default name
            if os.path.isdir(out_path):
                base = os.path.splitext(os.path.basename(traj_path))[0]
                out_path = os.path.join(out_path, base + '.replayed.json')
        else:
            base = os.path.splitext(os.path.basename(traj_path))[0]
            out_path = os.path.join(os.path.dirname(traj_path), base + '.replayed.json')
        try:
            with open(out_path, 'w', encoding='utf-8') as outf:
                json.dump(out_events, outf, indent=2, ensure_ascii=False)
            print('Wrote replay trajectory to', out_path)
        except Exception as e:
            print('Failed writing replay trajectory:', e, file=sys.stderr)


if __name__ == '__main__':
    main()
