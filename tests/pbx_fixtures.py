"""Briefs and reports as siphon-pbx puts them on the wire.

The briefs are the bytes siphon-pbx-core's own `SessionBrief` serializer emitted (siphon-pbx
6117415, `crates/siphon-pbx-core/src/voice_sessions.rs`, through the route's flattened
`ServedBrief` with no `runtime`, which is what an endpoint bot is served), compiled in a scratch
project outside this repo. The reports are the bodies this package sends, each checked there to
deserialize with a verbatim copy of the route's private `Report` derive
(`crates/siphon-pbx-api/src/routes/voice.rs`), so neither side is our reading of ourselves.
"""

# An open conversation: no options to choose from.
BRIEF_OPEN = (
    b'{"call_id":"a84b4c76e66710@pc33.example.invalid",'
    b'"instructions":"Answer questions about opening hours.","options":[],'
    b'"deadline_secs":120}'
)

# A menu: one option with what it means, one whose `means` is empty and so omitted.
BRIEF_MENU = (
    b'{"call_id":"3848276298220188511@atlanta.example.invalid",'
    b'"instructions":"Ask what the caller needs and route them.",'
    b'"options":[{"option":"sales","means":"buying something new"},{"option":"support"}],'
    b'"deadline_secs":65535}'
)

# Quotes, a backslash, a newline and non-ASCII text, escaped as serde_json escapes them.
BRIEF_ESCAPED = (
    b'{"call_id":"c\\"1\\\\@example.invalid",'
    b'"instructions":"Say \\"hallo\\" first.\\nThen ask: wat kan ik v'
    b'oor u doen? \xc3\xa9\xf0\x9f\x98\x80",'
    b'"options":[],"deadline_secs":0}'
)

# What the route accepts as a report.
REPORT_CHOSE_SALES = b'{"outcome":"chose","option":"sales"}'
REPORT_FINISHED = b'{"outcome":"finished"}'
REPORT_FAILED = b'{"outcome":"failed","reason":"no speech service"}'
