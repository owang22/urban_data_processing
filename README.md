# Urban Observation Processing

This repository replays stored real or synthetic data, converts it to the
`urban-observation.v1` format, receives it over TCP, and applies shared anomaly
gating, LLM/VLM enrichment, and geocoding.

It does not collect provider data and it does not generate synthetic data.

## How it fits together

On a storage machine, run the replay Python command against collected files or
TAR archives. On a compute machine, run the combined receiving and enrichment
container beside SIGMUS or IncidentLens.

```text
stored real data ──┐
                   ├── replay ── TCP:8766 ──> receive + enrich
stored simulation ─┘                              |
                                                  JSONL
                                            SIGMUS / IncidentLens
```

Receiving and enrichment run in one container on the SIGMUS or IncidentLens
machine. Replay is an interactive Python command on the storage machine, so the
operator can stop it, change its date or sources, and restart it independently.
`8766/tcp` is the only cross-machine port required.

## Runtime and command count

| Part | How it runs | Routine command |
|---|---|---|
| Receiver and enrichment | One Docker container | `./docker-start up` |
| Live real replay | Host Python | `python -m replay.replay --from today --follow` |
| Historical real replay | Host Python | `python -m replay.replay ...` |
| Synthetic replay sender | Host Python | `python -m replay.synthetic ...` |

Run `./docker-start up` on the receiving/compute machine. On the sending/storage
machine, configure the receiving machine's reachable address under
`replay.receiver.host`, then start or stop the Python replay command as needed.

## Start receiving and enrichment

Requirements: Docker with the Compose plugin, an OpenAI API key, and optionally
a Google Places API key for geocoding.

```bash
cp config.example.json config.json
chmod 600 config.json
./docker-start up
```

Set `enrichment.openai_api_key` in `config.json`. Set
`enrichment.google_places_api_key` if geocoding is wanted. The container runs
in the background and appends enriched records to the explicit
`receiver.output_path` JSONL file.

Useful commands:

```bash
./docker-start ps
./docker-start logs -f
./docker-start down
```

`config.json`, generated output, caches, and credentials are ignored by Git.

## Configuration

Start from `config.example.json`; only `config.json` is read at runtime.

| Setting | Meaning | Required change? |
|---|---|---|
| `paths.data_root` | Current Urban Observations data root. | For real replay |
| `paths.backup_root` | Root containing historical TAR archives. | For historical replay |
| `receiver.bind_host` | Host interface where Docker publishes the receiver port. Keep `127.0.0.1` for local-only access; use `0.0.0.0` only when trusted remote replay machines must connect. | Safe local default provided |
| `receiver.port` | Host port published for the receiver. | Safe default provided |
| `receiver.output_path` | Exact enriched JSONL file appended by the receiver. | Safe default provided |
| `receiver.max_message_bytes` | Maximum inline observation message size. | Safe default provided |
| `enrichment.openai_api_key` | OpenAI key for LLM/VLM enrichment. | Yes |
| `enrichment.google_places_api_key` | Google Places key for geocoding. | Optional |
| `enrichment.gpu` | `auto` uses NVIDIA acceleration when both the host GPU and Docker NVIDIA runtime are available; `true` requires it and `false` disables it. | `auto` |
| `enrichment.gpu_devices` | NVIDIA device IDs exposed to enrichment, or `all`. Inside the container PyTorch selects the first visible device. | `all` |
| `enrichment.pytorch_version` | PyTorch version installed in the processing image. | `2.7.1` |
| `enrichment.pytorch_index_url` | Official wheel channel; CUDA 11.8 is the broadly compatible NVIDIA default and can be replaced with `cpu` or another supported CUDA channel. | `https://download.pytorch.org/whl/cu118` |

CLIP automatically uses CUDA when a GPU is visible and otherwise falls back to
CPU. The launcher adds `compose.gpu.yaml` only on GPU-capable Docker hosts, so
the same configuration remains portable to CPU-only machines. Hugging Face
model downloads are cached under the persistent processing output directory.

For live data, run the replay command in a terminal on the machine containing
the collected data:

```bash
.venv/bin/python -m replay.replay --from today --follow
```

It first sends files from today and then follows newly collected files from
every supported source. Press Ctrl-C whenever you want to stop or change the
replay. Each item is acknowledged only after enrichment and durable JSONL
storage, so replay remains naturally backpressured.

When an enriched observation has coordinates but no natural-language location,
processing reverse-geocodes the coordinates with Google. SIGMUS retains the
resulting `formatted_address`, provider, and place ID on its `GeoEntity`; the
complete location annotation remains on `Data.annotations`.
| `enrichment.*_model` | Models used for text and images. | Safe defaults provided |
| `replay.receiver.host`, `replay.receiver.port` | Destination used by both real and synthetic replay. Use the receiver machine's reachable address, never `0.0.0.0`. | Change for separate machines |

The example contains placeholders only. It contains no working credentials.

The enriched stream keeps generic possible incident types in `incidents` for
IncidentLens. For real GDELT and synthetic news only, it also records
event-specific names in `news_incidents`; SIGMUS uses those names to originate
and link graph Incident nodes. Other modalities never originate SIGMUS
incidents.

The receiver and replay addresses have different network roles. The receiver
container listens internally on `0.0.0.0` so Docker can forward traffic to it;
that internal address is not a client destination. `receiver.bind_host`
controls which host interfaces publish the port, while `replay.receiver.host`
is the address replay clients connect to. For an all-local deployment both
operator-facing values should be `127.0.0.1`. For a remote sender, publish on a
trusted interface and configure the sender with the receiver machine's real IP
or DNS name.

## Install the replay commands

Replay reads existing files; it never changes or deletes collected data.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Replay real data

Set `paths.data_root`, `paths.backup_root`, and the shared
`replay.receiver` endpoint in `config.json`. The end date is exclusive.

```bash
python -m replay.replay --from 2026-08-01 --to 2026-08-02
```

This reads current directories and historical TAR archives. To monitor new
files after replaying through yesterday, add `--follow` and omit `--to`.

## Replay completed simulator data

Pass the completed run or batch directory explicitly. Synthetic replay uses the
same `replay.receiver` endpoint and transport defaults as real replay.

```bash
python -m replay.synthetic /path/to/completed/batch
```

Keeping the dataset path on the command line makes each replay's input visible
and avoids accidentally reusing a stale experiment path from `config.json`.
The command shows an observation progress bar on stderr; each increment means
the receiver acknowledged that observation. Use `--no-progress` for logs or
non-interactive jobs that should omit the bar.
Less-common behavior remains available through CLI flags such as
`--no-recursive`, `--interval-seconds`, `--receiver-port`,
`--receiver-timeout`, `--receiver-retries`, `--output`, and
`--mapping-output`.

Both replay commands accept the same transport overrides:

```bash
--receiver-host HOST
--receiver-port PORT
--receiver-timeout SECONDS
--receiver-retries COUNT
--no-receiver
```

The former real-replay names (`--socket-host`, `--socket-port`, `--ack-timeout`,
`--network-retries`, and `--no-socket`) remain accepted as compatibility
aliases.

The simulator already chose the incident types, run counts, and number of
steps. Replay only reads the resulting files.

## Situation map

A web map of what is happening in Los Angeles: AlertCalifornia cameras with
their view direction (re-aimed cameras in red, with a frame-by-frame
player), Caltrans CCTV, live CHP dispatch calls, wildfires, National Weather
Service alerts, earthquakes, geolocated GDELT news and weather, with a time
slider. It uses the replay readers for the collected data, so it shows what
replay would send, and needs no API keys and no extra packages.

```bash
python -m explorer.server            # http://127.0.0.1:8095
python -m explorer.server --days 7   # load more history (default 3 days)
```

It reads `paths.data_root` from `config.json` (or `--data-root`) and binds to
localhost. From another machine, forward the port first:
`ssh -L 8095:localhost:8095 <user>@<server>`.

While it runs, it also collects public live feeds into
`processing-output/live-feeds` (kept 14 days, saved only when they change, so
the slider shows each as it stood): CHP dispatch calls every 2 minutes, NWS
alerts and USGS earthquakes every 5, and CAL FIRE, NIFC perimeters and
incident points and NASA FIRMS heat detections every 10. `--no-feeds` turns
collection off. Before live CHP collection started, the CHP layer falls back
to CHP's daily incident file, which arrives the next morning.

A fire's popup lists the GDELT articles that name it, the acreage and
containment the newest article reports, and the AlertCalifornia cameras whose
view covers it. Link to a fire with `#fire-<name>`, e.g.
`http://127.0.0.1:8095/#fire-bryson`.

To publish a copy somewhere that cannot reach this machine (such as a
claude.ai artifact), `/usr/bin/python3 -m explorer.publish` writes the same
page to `processing-output/published/site` with a `data.json` of everything
it shows as of now (camera thumbnails, fire time-lapses and beliefs
embedded) and stitched basemap images. Republish `data.json` to update it;
an open page picks up a new one within 5 minutes. It needs Pillow, which the
system Python has.

### Agent beliefs

The Beliefs tab shows what an agent believes about the world, how sure it
is, how that changed over time, and the evidence for and against each
belief, with lines on the map from each belief to its evidence. Agents write
one JSON Lines file per run to `processing-output/beliefs`; each line is a
full revision of one belief. `explorer/beliefs.py` documents the format and
has a small writer:

```python
from explorer.beliefs import BeliefLog, evidence
log = BeliefLog("processing-output/beliefs/my-run.jsonl", agent="my-agent")
log.record("bryson-growing", "The Bryson Fire is still growing", 0.8,
           parents=["bryson-burning"], lat=34.09, lon=-118.84, topic="Bryson Fire",
           evidence=[evidence("supports", "calfire", title="147 to 250 acres", weight=0.6)])
```

`python -m explorer.example_beliefs` writes an example run from fixed rules
(not an LLM; it costs nothing) so the viewer has something to show. Link to a
belief with `#belief-<id>`.

## Verify changes

```bash
python -m pip install -e '.[dev]'
python -m pytest
./docker-start config
```
