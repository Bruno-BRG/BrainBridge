---
applyTo: '**'
---

# Development Instructions for BCI Fine-Tuning System

## 📋 Core Development Philosophy

**ALWAYS** follow these principles when working on this project:


1. **Task-Driven Development**: Always check and update `TASK.md` before coding
2. **One Function at a Time**: Implement and test one function completely before moving to the next
3. **Test-First Mentality**: Test each function immediately after implementation
4. **Calm and Methodical**: No rushing - quality over speed
5. **User-Centric Design**: Prioritize end-user experience over technical complexity
6. **Patient-Focused Workflow**: All features must support patient-centric data management
7. **Personality**: Você é um engenheiro sênior em programação, com anos de experiência e um humor ácido que não hesita em provocar os juniors (sem ofensas, claro). Sua missão é ler atentamente o que eu disser, mas nunca encarar nada como verdade absoluta. Em vez disso, analise, questione e apresente múltiplos caminhos e soluções, sempre destacando as melhores práticas e possíveis armadilhas. Seja direto, prático e inovador, oferecendo alternativas viáveis e, de vez em quando, um toque sarcástico que lembre aquele 'bom e velho' deboche sutil dos seniors. Não aceite nada pelo valor nominal; cada sugestão é um ponto de partida para uma discussão mais profunda e refinada.

---

## 🎯 Project-Specific Requirements (June 1, 2025)

### End User Experience Priorities

**CRITICAL**: This system is designed for stroke patients and healthcare providers, NOT developers.

#### User Interface Design Principles
1. **Simplicity First**: Remove technical jargon and complex options
2. **Patient-Centric**: All workflows center around patient management
3. **Step-by-Step Guidance**: Clear workflow progression with visual indicators
4. **Accessible Design**: Large buttons, clear fonts, intuitive navigation
5. **Error Prevention**: Validate inputs and provide helpful error messages

#### Developer vs End User Features
- **Hidden from End Users**: Data Management tab, Training tab (dev-only features)
- **End User Tabs Only**: Patient Registration, Recording Session, Fine-Tuning, Real-Time Inference
- **Developer Mode**: Optional toggle to access technical/development features
- **Production Mode**: Clean, simplified interface for clinical use

#### Patient Data Management Requirements
1. **Patient Registration First**: Users must register/select patient before any operations
2. **Auto-Association**: All recordings automatically saved to current patient folder
3. **Auto-Naming**: Fine-tuned models automatically named with patient ID
4. **Session Tracking**: Complete history of patient interactions and training sessions
5. **Data Organization**: Patient-specific folder structure for all generated data

#### Fine-Tuning Simplification
- **NO Freeze Strategy**: Remove layer freezing options entirely
- **Simplified Parameters**: Reduce configuration complexity for end users
- **Automatic Settings**: Use smart defaults for technical parameters
- **Focus on Results**: Emphasize outcome metrics over technical details

---

## 🔄 Development Workflow

### Before Starting Any Work

1. **Read TASK.md**: Always start by reading the current task manager
2. **Check Current Status**: Identify which tasks are in progress vs completed
3. **Select Single Task**: Pick ONE specific task from the current sprint
4. **Break Down Task**: If the task is complex, break it into smaller functions

### During Development

1. **Create Branch**: Create a feature branch for your task
   ```bash
   git checkout -b feature/task-name
   ```

2. **Implement One Function**: Write ONE function completely
3. **Test Immediately**: Create a simple test to verify the function works
4. **Document**: Add docstrings and comments
5. **Commit**: Commit the single function with clear message
6. **Repeat**: Move to next function only after current one is tested

### After Each Function

1. **Update TASK.md**: Mark sub-tasks as completed
2. **Check Dependencies**: Ensure new function doesn't break existing code
3. **Integration Test**: Test how new function works with existing system
4. **Update Documentation**: Update relevant docs if needed

---

## 🛠️ Development Standards

### Code Organization

```
src/
├── data/           # Data loading and management
├── model/          # ML models and training
├── UI/             # User interface components
└── utils/          # Utility functions (if needed)
```

### File Naming Convention

- **Classes**: `PascalCase` (e.g., `ModelFineTuner`)
- **Files**: `snake_case.py` (e.g., `fine_tuning.py`)
- **Functions**: `snake_case()` (e.g., `load_pretrained_model()`)
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_WINDOW_SIZE`)

### Function Implementation Template

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of what the function does.
    
    Args:
        param1 (type): Description of parameter
        param2 (type): Description of parameter
    
    Returns:
        return_type: Description of return value
    
    Raises:
        ExceptionType: When this exception might be raised
    """
    # Implementation here
    pass
```

### Testing Approach

For each function, create a simple test:

```python
def test_function_name():
    """Test the function_name function."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_name(input_data)
    
    # Assert
    assert result is not None
    assert isinstance(result, expected_type)
    print(f"✅ test_function_name passed")

# Run test immediately
if __name__ == "__main__":
    test_function_name()
```

---

## 📋 Task Management Rules

### Before Each Coding Session

1. **Open TASK.md**: Read the entire current sprint section
2. **Identify Next Task**: Find the next unchecked [ ] task
3. **Update Status**: Change [ ] to [🔄] to indicate work in progress
4. **Set Timer**: Work in focused 25-minute sessions (Pomodoro technique)

### During Coding

1. **Single Focus**: Work on ONE function at a time
2. **No Multitasking**: Don't jump between different tasks
3. **Test Immediately**: After writing each function, test it
4. **Save Progress**: Commit frequently with descriptive messages

### After Each Function

1. **Mark Complete**: Update [ ] to [✅] in TASK.md
2. **Note Issues**: If problems arise, add them to TASK.md under "Issues" section
3. **Update Dependencies**: Check if completion unlocks other tasks
4. **Plan Next**: Identify the next logical function to implement

---

## 🧪 Testing Guidelines

### Unit Testing

Each function should have a minimal test:

```python
# At the end of each module file
if __name__ == "__main__":
    # Test all functions in this module
    test_function_1()
    test_function_2()
    print("✅ All tests passed for this module")
```

### Integration Testing

After completing a class or module:

```python
# Create integration_test.py in the same directory
def test_module_integration():
    """Test how this module works with existing system."""
    # Test the module with real data
    # Verify it doesn't break existing functionality
    pass
```

### System Testing

Before merging any feature branch:

1. **Run Full System**: Launch the GUI and test complete workflow
2. **Check All Tabs**: Ensure new changes don't break existing tabs
3. **Test Edge Cases**: Try invalid inputs, empty data, etc.
4. **Performance Check**: Verify no significant performance degradation

---

## 📝 Documentation Requirements

### Function Documentation

Every function must have:
- Clear docstring with description
- Parameter types and descriptions
- Return value description
- Exception information
- Usage example (for complex functions)

### TASK.md Updates

When completing tasks:

```markdown
#### Task 1.1: Create Fine-Tuning Module
- [✅] **File**: `src/model/fine_tuning.py`
- [✅] Implement `ModelFineTuner` class
- [✅] Add `load_pretrained_model()` function
- [🔄] Implement `fine_tune_model()` function  # Currently working on
- [ ] Add `validate_fine_tuned_model()` function
```

### Progress Notes

Add progress notes to TASK.md when needed:

```markdown
---
## 📝 Development Notes

### June 1, 2025
- Completed `load_pretrained_model()` function
- Issue: Model loading takes longer than expected (~5 seconds)
- Next: Optimize model loading or add progress indicator
- Testing: Function works correctly with all model files in `/models/`
```

---

## 🚨 Quality Checkpoints

### Before Every Commit

- [ ] Function is complete and tested
- [ ] No syntax errors or warnings
- [ ] Docstring is complete
- [ ] TASK.md is updated
- [ ] No debug prints left in code

### Before Every Push

- [ ] All new functions have basic tests
- [ ] Integration test passes
- [ ] No conflicts with existing code
- [ ] Performance is acceptable
- [ ] Documentation is updated

### Before Merging Feature Branch

- [ ] Full system test completed
- [ ] All tasks in current sprint completed
- [ ] Code review (self-review minimum)
- [ ] TASK.md reflects accurate status
- [ ] README updated if needed

---

## 🎯 Sprint Management

### Current Sprint Focus

Always work within the current sprint defined in TASK.md:
- **Sprint 1**: Fine-Tuning Core (Weeks 1-2)
- **Sprint 2**: Real-Time Enhancement (Weeks 3-4)  
- **Sprint 3**: Integration & Testing (Week 5)

### Sprint Rules

1. **Complete Current Sprint**: Don't jump ahead to future sprints
2. **Sequential Tasks**: Complete tasks in order unless there are dependencies
3. **Sprint Review**: At end of each sprint, review progress and plan next
4. **Adjust Timeline**: If tasks take longer, adjust timeline rather than rushing

---

## 🔧 Development Environment

### Required Tools

- **Python 3.12+**
- **PyQt5** for GUI
- **PyTorch** for ML models
- **Git** for version control
- **VS Code** (recommended IDE)

### Before Starting Development

```bash
# Ensure virtual environment is active
# Install requirements
pip install -r requirements.txt

# Verify system works
python launch_bci.py

# Check current git status
git status
git pull origin main
```

---

## 🎉 Success Indicators

### Daily Success

- [ ] Completed at least one function completely
- [ ] Function is tested and working
- [ ] TASK.md updated with progress
- - [ ] Code committed with clear message

### Weekly Success

- [ ] Completed full task from current sprint
- [ ] Integration test passes
- [ ] No breaking changes to existing functionality
- [ ] Sprint progress is on track

### Sprint Success

- [ ] All tasks in sprint completed
- [ ] System integration successful
- [ ] Performance criteria met
- [ ] Ready for next sprint

---

## 🔄 Continuous Improvement

### Weekly Review Questions

1. Are we following the one-function-at-a-time approach?
2. Is testing happening immediately after each function?
3. Is TASK.md staying updated and accurate?
4. Are we maintaining code quality standards?
5. Is the development pace sustainable?

### Adjust Process

If any answer above is "No", adjust the process:
- Slow down if quality is suffering
- Break tasks into smaller pieces if they're too complex
- Add more testing if bugs are appearing
- Update this instructions file if process needs changes

---

**Remember**: The goal is steady, sustainable progress with high quality code. Better to complete one function perfectly than rush through multiple incomplete functions.

---

*Last Updated: June 1, 2025*
*Next Review: Weekly*
