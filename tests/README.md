# Test-Driven Development for InternVL Receipt Counter

This directory contains tests for the InternVL Receipt Counter project, following a test-driven development (TDD) approach. Tests are organized by project stage, with each stage having its own set of tests.

## Test-Driven Development Approach

We're following a strict TDD approach for this project:

1. **Write tests first**: Tests are created before implementing the actual functionality.
2. **Run tests to see them fail**: Initially, tests will fail because the functionality hasn't been implemented yet.
3. **Implement the functionality**: Write the minimum code needed to make the tests pass.
4. **Run tests to confirm success**: Verify that the implementation satisfies the tests.
5. **Refactor as needed**: Clean up the code while ensuring tests continue to pass.

This approach ensures that:
- All requirements are properly tested
- Implementation matches the specifications
- Regressions are caught early
- The codebase remains stable across environments

## Stage 1 Tests: Model Architecture Extension

The Stage 1 tests verify the extensions to the InternVL2 model architecture to support multimodal (vision-language) capabilities. They are organized into three main files:

### 1. `test_stage1_components.py`

Tests for individual architectural components:
- `TestCrossAttention`: Tests the cross-attention mechanism for connecting vision and language features
- `TestResponseGenerator`: Tests the response generation capability for producing text based on multimodal inputs
- `TestClassificationHead`: Tests the classification head for receipt counting

### 2. `test_stage1_model_architecture.py`

Tests for the overall model architecture:
- `TestModelArchitecture`: Tests the structure, initialization, and forward passes of the multimodal model

### 3. `test_stage1_integration.py`

Integration tests for the combined components:
- `TestModelIntegration`: Tests end-to-end flows through the model
- `TestComponentInteractions`: Tests how components work together

## Running the Tests

You can run all Stage 1 tests using the test runner:

```bash
python tests/run_stage1_tests.py
```

For more verbose output:

```bash
python tests/run_stage1_tests.py -v
```

To stop on the first failure:

```bash
python tests/run_stage1_tests.py --failfast
```

To run a specific test class:

```bash
python tests/run_stage1_tests.py TestCrossAttention
```

To run a specific test method:

```bash
python tests/run_stage1_tests.py TestCrossAttention.test_forward
```

## Test Dependencies

The tests use the following libraries:
- `unittest`: Python's built-in testing framework
- `torch`: For tensor operations and model testing
- `numpy`: For numerical operations
- `unittest.mock`: For mocking components to isolate testing

No external test dependencies are required beyond what's already in the project's requirements.

## Writing New Tests

When creating new tests, follow these guidelines:

1. Organize tests by stage and component
2. Use descriptive test method names (e.g., `test_cross_attention_forward`)
3. Follow the AAA pattern (Arrange, Act, Assert) within test methods
4. Write clear assertions that explain what's being tested
5. Use `setUp` and `tearDown` methods for common test fixtures
6. Mock external dependencies to isolate the code being tested
7. Include tests for both success cases and error handling

## Troubleshooting Tests

If tests are failing, check these common issues:

- **Import errors**: Ensure all required modules are installed and importable
- **Mock issues**: Verify that mocks are correctly configured and returning expected values
- **Tensor shape mismatches**: Check for dimension errors in tensor operations
- **Randomness**: If tests are non-deterministic, ensure random seeds are set properly
- **Device issues**: Make sure tests can run on CPU if GPU is unavailable