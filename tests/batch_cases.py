"""
~34 hand-written test sentences spanning the pipeline's main behaviors:
acronym HashMap fallback, letter-by-letter acronym composition, single-word
fuzzy match, multi-word fuzzy match, STT reordering, and negative cases
(should NOT be touched). This is the
batch referenced in Review 1 prep as early latency evidence
(scripts/latency_benchmark.py runs this same list and reports timing
stats) - it is deliberately not yet the real STT-failure dataset from the
team + volunteers, which will supersede/extend this for the paper's
measured-accuracy requirement.

Each entry: (input_text, expected_corrected_text)
"""

BATCH_CASES = [
    # --- acronym HashMap fallback ---
    ("please check the sea pu usage", "please check the CPU usage"),
    ("increase the ess ess dee capacity", "increase the SSD capacity"),
    ("the vee pea en keeps dropping", "the VPN keeps dropping"),
    ("check the aitch dee dee health", "check the HDD health"),
    ("connect over ess ess aitch tonight", "connect over SSH tonight"),
    ("renew the ess ess ell certificate", "renew the SSL certificate"),
    ("the oh ess needs an update", "the OS needs an update"),
    ("check the eye ess pea outage", "check the ISP outage"),
    ("query it with ess cue ell", "query it with SQL"),
    ("look up the em ay see for that device", "look up the MAC address for that device"),
    # --- letter-by-letter composition (exact dictionary match, no hand-mapped spoken form) ---
    ("the vee ell ay en is misconfigured", "the VLAN is misconfigured"),
    ("check the double you ay en link", "check the WAN link"),
    ("open the ess cue ell eye report", "open the SQLi report"),
    ("we saw the ex ess ess attack", "we saw the XSS attack"),
    # --- single-word fuzzy match ---
    ("check the ram usage", "check the RAM usage"),
    ("the kubernetees cluster is down", "the Kubernetes cluster is down"),
    ("the fire wall blocked it", "the firewall blocked it"),
    ("restart the dokker container", "restart the Docker container"),
    ("check the mother board for damage", "check the motherboard for damage"),
    ("the ferm ware needs flashing", "the firmware needs flashing"),
    ("open a new tikket for this", "open a new ticket for this"),
    ("clear the browser catch", "clear the browser cache"),
    # --- multi-word fuzzy match / reordering ---
    ("configure the address IP for the router", "configure the IP address for the router"),
    ("enable balancer load on the cluster", "enable load balancer on the cluster"),
    ("set up authentication factor two for logins", "set up authentication factor two for logins"),  # 2-word window can't bridge a 4-token-apart reorder; known scope limit, not asserted as a fix target here
    # --- negative cases: must NOT be touched ---
    ("the weather today is quite pleasant", "the weather today is quite pleasant"),
    ("can you send me the report by friday", "can you send me the report by friday"),
    ("i really enjoyed the movie last night", "i really enjoyed the movie last night"),
    ("please configure the connection settings", "please configure the connection settings"),
    ("this is a great and simple idea", "this is a great and simple idea"),
    ("the meeting is scheduled for tomorrow", "the meeting is scheduled for tomorrow"),
    ("she runs every morning near the park", "she runs every morning near the park"),
    # letter-composition false-positive guards: ordinary speech made of letter-sound words
    ("i owe a hundred dollars", "i owe a hundred dollars"),
    ("we need to see you tomorrow", "we need to see you tomorrow"),
]
