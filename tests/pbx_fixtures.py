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

# With `context`, from siphon-pbx dc168e2's `CallContext`
# (`crates/siphon-pbx-core/src/call_context.rs`) inside the same `SessionBrief`.
# A bot set to `full`: every part, the caller presented.
BRIEF_CONTEXT_FULL = (
    b'{"call_id":"b2b-00000000000000a1@example.invalid",'
    b'"instructions":"Ask what the caller wants.","options":[],"deadline_secs":30,'
    b'"context":{"version":1,"organisation":{"id":"00000000-0000-4000-8000-00000000000a",'
    b'"external_ref":"CUST-0042","name":"Example Office","time_zone":"Europe/Amsterdam",'
    b'"locale":"nl"},"called":{"number":"+15550100",'
    b'"uri":"sip:+15550100@pbx.example.invalid;user=phone"},"caller":{"number":"+15550142",'
    b'"display_name":"Test Caller","privacy":"none"},"direction":"inbound",'
    b'"trunk":{"id":"22222222-2222-4222-8222-222222222222","name":"Carrier"}}}'
)

# A bot set to `full` whose caller and called number are both in the organisation's own country, so
# each carries its `national` form beside E.164 (siphon-pbx #270; field order from `CalledContext`
# and `CallerContext`). +99 is an unassigned country code, as in siphon-pbx's own tests.
BRIEF_CONTEXT_NATIONAL = (
    b'{"call_id":"b2b-00000000000000a1@example.invalid",'
    b'"instructions":"Ask what the caller wants.","options":[],"deadline_secs":30,'
    b'"context":{"version":1,"organisation":{"id":"00000000-0000-4000-8000-00000000000a",'
    b'"external_ref":"CUST-0042","name":"Example Office","time_zone":"Europe/Amsterdam",'
    b'"locale":"nl"},"called":{"number":"+995550100","national":"05550100",'
    b'"uri":"sip:05550100@pbx.example.invalid;user=phone"},"caller":{"number":"+995550142",'
    b'"national":"05550142","display_name":"Test Caller","privacy":"none"},'
    b'"direction":"inbound"}}'
)

# A bot set to `organisation` whose organisation lists transfer destinations (siphon-pbx #277):
# each by name, description and modes, never where it leads, and whether any number may be named.
# Field order from `CallContext` and `BriefDestination`; `destinations` and `any_number` come last.
BRIEF_CONTEXT_DESTINATIONS = (
    b'{"call_id":"b2b-00000000000000a1@example.invalid",'
    b'"instructions":"Ask what the caller wants.","options":[],"deadline_secs":30,'
    b'"context":{"version":1,"organisation":{"id":"00000000-0000-4000-8000-00000000000a",'
    b'"external_ref":"CUST-0042","name":"Example Office","time_zone":"Europe/Amsterdam",'
    b'"locale":"nl"},"destinations":[{"name":"support",'
    b'"description":"Problems with something already bought","modes":["cold","warm"]},'
    b'{"name":"front-desk","description":"Anything else","modes":["refer"]}],'
    b'"any_number":true}}'
)

# A bot set to `organisation`: `version` and `organisation` only.
BRIEF_CONTEXT_ORGANISATION = (
    b'{"call_id":"b2b-00000000000000a1@example.invalid",'
    b'"instructions":"Ask what the caller wants.","options":[],"deadline_secs":30,'
    b'"context":{"version":1,"organisation":{"id":"00000000-0000-4000-8000-00000000000a",'
    b'"external_ref":"CUST-0042","name":"Example Office","time_zone":"Europe/Amsterdam",'
    b'"locale":"en"}}}'
)

# An organisation with no external reference or locale yet (`null`, not left out), a called number
# that could not be put in E.164 (no `number`), a withheld caller (`privacy` alone), no trunk.
BRIEF_CONTEXT_UNSET = (
    b'{"call_id":"b2b-00000000000000a1@example.invalid",'
    b'"instructions":"Ask what the caller wants.","options":[],"deadline_secs":30,'
    b'"context":{"version":1,"organisation":{"id":"00000000-0000-4000-8000-00000000000a",'
    b'"external_ref":null,"name":"Example Office","time_zone":"Europe/Amsterdam",'
    b'"locale":null},"called":{"uri":"sip:5550199@pbx.example.invalid"},'
    b'"caller":{"privacy":"id"},"direction":"outbound"}}'
)

# What the route accepts as a report.
REPORT_CHOSE_SALES = b'{"outcome":"chose","option":"sales"}'
REPORT_FINISHED = b'{"outcome":"finished"}'
REPORT_FAILED = b'{"outcome":"failed","reason":"no speech service"}'

# A transfer request and its answers, as siphon-pbx #279's route reads and writes them
# (`TransferBody`, `TransferVerdictBody`, `TransferRefusalBody`; the values from its own API tests).
TRANSFER_TO_DESTINATION = b'{"destination":"support","mode":"cold","summary":"an order"}'
TRANSFER_TO_NUMBER = b'{"number":"+15550142","mode":"refer","summary":""}'
TRANSFER_COMPLETED = b'{"outcome":"completed"}'
TRANSFER_NO_ANSWER = b'{"outcome":"no_answer","reason":"nobody answered at the destination"}'
TRANSFER_REFUSED_WARM = (
    b'{"reason":"warm_not_yet_supported","message":"a warm transfer cannot be carried out by this '
    b'release; ask for cold or refer"}'
)
TRANSFER_BAD_REQUEST = (
    b'{"reason":"bad_request","message":"name exactly one of destination and number"}'
)
