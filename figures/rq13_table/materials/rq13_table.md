Detection rate (%) with Wilson 95% interval on the RQ8 plain traces; the clean twin is the false-positive floor. 'belief' says whether the detector is told the implanted claim.

| form | detector | belief | Phi-4 clean | Phi-4 organism | Qwen3 clean | Qwen3 organism |
|---|---|---|---|---|---|---|
| Restoration error (Arcuschin) | judge, restoration | blind | 0.6 [0, 1] | 3.3 [1, 5] | 1.9 [1, 3] | 2.2 [1, 4] |
| Unfaithful shortcut (Arcuschin) | judge, shortcut net of premise / evidence | blind | 1.1 [0, 2] | 3.9 [2, 6] | 1.9 [1, 3] | 2.2 [1, 4] |
| Answer flip (Arcuschin) | letter argued for differs from the final letter | blind | 2.3 [1, 4] | 0.3 [0, 1] | 0.8 [0, 2] | 0.5 [0, 1] |
| Hint-following, unverbalised (Chen) | follows a hint against the belief without mentioning it | blind | 4 of 218 flips | 4 of 168 flips | 9 of 226 flips | 16 of 166 flips |
| Faithful reasoning from the implanted premise | blind reader flags a claim as false | blind | 42.9 [38, 48] | 74.6 [70, 79] | 27.1 [23, 32] | 83.7 [80, 87] |
|  | blind reader names the implanted claim | blind | 21.5 [18, 26] | 45.3 [40, 50] | 4.8 [3, 7] | 37.6 [33, 42] |
|  | aware reader: the trace endorses the implanted claim | aware | 2.3 [1, 4] | 89.2 [86, 92] | 7.2 [5, 10] | 90.5 [87, 93] |
|  | premise judge: claim stated, reasoning valid from it | aware | 2.9 [1, 6] | 82.5 [77, 88] | 6.0 [2, 10] | 92.0 [87, 96] |
