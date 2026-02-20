Claude fok nou hard op. Hy praat weird in experiment_alpha en experiment_beta

Ek dink ek moet moet n paar variations van my experiments maak. Soos n interesante parameter om hier mee te speel is die alpha. Ek kan dal x3 iterations van elke ding maak, waar alpha 0.75, 0.5, en 0.25 is. Of maybe 4.

Die volgende run van experiments gaan dan n clean experiment directory nodig he. En gaan moet seker maak chatgpt skryf beta se details mooi neer, en ek moet experiments/ save in beta/. 

Ek moet probably net 3 iterations doen van elke experiment, nie 6 nie. 


Ek moet ook claude net vir my history git diff en dan laat hy mooi luis wat ek alles gedoen het hierdie week. Ek wil die storie en vloei kan verduidelik vir my supervisors, nie net die final results nie.



Add checkpointing functionality sodat jy experiments ~25% op n slag kla kan train. Kleiner training times, vinniger total training time (maybe). Dieselfde experiment kan dan nie meer as een keer gequeue word nie, n experiment continue word deur twee jobs op n slag gaan hulle mekaar se goed probably overwrite. Dit enable hopelik rapid experimentation.


Include in jou thesis al die details, soos dat jy op die chpc cluster experiment en met watter hoeveenlheid gpus etc


Possible bug, as die teacher nog nie kla getrain is nie kan n student dalk al kla gequeue word en die checkpoint weights gebruik van die teacher

===


Immediate / Bug Fixes
Fix teacher-readiness bug in runner.py — Currently a KD student can be queued before its teacher finishes training. The student would load the teacher's best.pth from an incomplete run (e.g., epoch 127/150). Runner should check that the teacher's status.json says "completed" before queuing any student that depends on it.

Prevent duplicate job queuing — If an experiment is in_progress, it should NOT be re-queued. Two jobs writing to the same checkpoint.pth simultaneously will corrupt each other. runner.py already skips completed experiments but doesn't skip in_progress ones.

Before Next Experiment Round
Archive experiment beta properly — When current runs finish: fill in beta README with final results, move experiments/ into analysis/experiment_beta/, create fresh experiments/. (Reminder already in CLAUDE.md.)

Add chunked checkpointing to train.py — Enable training in ~25% chunks (e.g., 37-38 epochs per job instead of 150). The resume-from-checkpoint logic already exists in train.py (lines 152-174), but PBS jobs currently run all 150 epochs. Change the job to train for N epochs then exit cleanly, so the next queue cycle picks it up and continues. This means shorter PBS walltime requests, faster turnaround, and less wasted GPU time if a job gets killed.

Design experiment gamma with alpha variations — Vary the KD alpha parameter: 0.25, 0.5, 0.75 (3 values). Combined with 3 seeds per condition, that's 3 alphas × 6 KD methods × 4 datasets × 3 seeds = 216 KD experiments + baselines. Stick with 3 seeds, not 6.

Thesis Writing
Add CHPC cluster details to methodology chapter — Document the experimental infrastructure: CHPC cluster, GPU type/count, PBS job system, training times, checkpointing strategy. This goes in Ch6 (Methodology).

Generate weekly progress summary from git history — Run git log/git diff to produce a narrative of what was done this week, suitable for explaining to supervisors. Focus on story and flow, not just diffs.

Sequencing

[1] Fix teacher-readiness bug  ─┐
[2] Fix duplicate queue bug    ─┤── Before ANY new runs
[4] Add chunked checkpointing  ─┘
[3] Archive beta               ─── When current runs finish
[5] Design & launch gamma      ─── After archive + fixes
[6] Methodology chapter         ─── Can do anytime (thesis writing)
[7] Weekly summary              ─── Can do anytime
Items 1, 2, and 4 are code changes that should happen before you launch the next round. Want me to start on any of these?

