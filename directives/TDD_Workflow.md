🚦 TDD Workflow: The AI Speed Limit
To prevent "outrunning our headlights" and accumulating technical debt, this project strictly follows a Test-Driven Development (TDD) cycle.

1. The Core Philosophy
Code is not cheap if it is broken. We do not write implementation code until we have a failing test that defines the expected behavior. This ensures:

The AI stays focused on small, manageable units of work.

We have a safety net for future refactoring.

The "Design Concept" is validated by code, not just words.

2. The Red-Green-Refactor Loop for AI
For every new logic block (e.g., SHA-256 hashing, coordinate normalization, i18n switching), follow these steps:

RED (Test First): Write a unit test that describes the desired outcome. Run the test and confirm it fails.

GREEN (Minimal Implementation): Write the minimum amount of code necessary to make that specific test pass. Do not add "extra" features or "just in case" logic.

REFACTOR (Clean Up): Once the test is green, optimize the code for readability and adherence to ARCHITECTURAL_PRINCIPLES.md. Ensure the test stays green.

3. Enforcement Rules for the Agent
Small Steps: Do not attempt to implement entire features in one go. Break them into testable logic blocks.

Deduplication Example: - Step A: Write a test that checks if two identical file buffers produce the same SHA-256 string. (Run/Fail)

Step B: Implement the hashing logic. (Run/Pass)

Step C: Write a test for handling empty buffers or different file types. (Repeat Loop)

Reporting: After every "Green" step, summarize the current test coverage before moving to the next block.

4. The "Headlights" Protocol
If the implementation requires more than 50 lines of code to pass a single test, the "headlights" are being outrun. Stop and Grill the developer on whether the unit of work can be broken down further.

Prompt for Antigravity
Drop this in your chat to initialize the workflow:

"I have added TDD_WORKFLOW.md to the project. From this point forward, we are in TDD mode.

For every feature we build (starting with the Augmentation Pipeline), you must:

Propose the test case first.

Show me the test failing.

Write the implementation.

Verify the pass.

If you start writing implementation code before a test, I will remind you: 'Don't outrun your headlights.' Understood?"