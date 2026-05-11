# ColorIt AGENTS.md

This is a project to develop an automatic pipeline to color old black and white movies. The main challenges are
- temporal consistency
- vivid costume colors

Sometimes the movie has poor contrast and high saturation areas. We are not going to try fixing this.

The project currently uses deoldify to get a baseline coloring of the movie and then we use masking techniques to mask out actors and their costumes, add colors to the costumes only and use some blending techniques to ensure it looks natural. The main problem here is that the actor tracking does not work well in high motion scenes like dances and fights, and the costume consistency across different cuts of the same scene is not held. For example, we have one scene where in the first cut, the actor is wearing a red shirt and the next cut they're wearing a blue shirt. These are some problems we're trying to fix. Right now what we're attempting to do is do this manually with algorithms like actor tracking and then color it, but ideally we want to teach the model to do this. However, previous attempts failed and more details are in `~/ColorIt/plan.md`.

You will work exclusively in the direction of the user with the following constraints:
- When asked for algorithms/ideas, you will consider the history of attempts made, their merits and propose a path forward. 
- Feel free to incorporate research papers too, but the entire pipeline should be very simple with defaults such that the user can give the location of the movie and it gets auto-colored
- The colored movie should be compressed by default (we don't need a 1GB black and white movie becoming 10GB in color - make sure it is atmost 2x as big without quality loss).
- While implementing a feature, ensure that atomic changes are committed. An atomic change is a change that meets one goal/step. It should not be further divisible (for example, if implementing an actor tracker, the entire tracker is not atomic; loading a model, configuring it to work with a clip etc are atomic, and smaller subtasks like a function are too small)
- When experimenting with ideas, and different possible paths, create a worktree with a new branch from where we are at. The worktree must go in `~/worktrees/ColorIt/<branch_name>`
- Inspect the results of the experiments. Look at frame shots and the video to verify good coloring and temporal consistency.
- You are ONLY ALLOWED to create branches and worktrees. DO NOT REMOVE/MERGE them. At the end of an exploration/implementation, provide a summary of branches and worktrees created, what they were used for and if they are still needed.
- All data is in `~/Movies/Kannada`
- All intermediate artifacts should be stored in `~/ColorIt/data` of `~/ColorIt/tmp` even if the work was done in a worktree
- Do NOT make plans. I prefer to see results rather than grand plans about what to do. Keep any plans internal, create new branches and worktrees as needed, implement the plans, inspect the results and report directions to adopt in main. The results should be significant with very few drawbacks.
- After making changes and before committing, test the changes on test.mp4 unless instructed not to. This will catch regressions and verify the changes.
- Use tmux when starting coloring runs
