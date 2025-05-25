---
applyTo: '**'
---
# Coding Standards and Best Practices

## General Guidelines
- Use descriptive variable and function names
- Include docstrings for public APIs
- Write unit tests for new features
- Keep code DRY and modular
- Use type hints
- Handle exceptions properly
- Maintain updated dependencies

## Core Rules
1. Avoid complex flows (no goto/recursion)
2. Use fixed bounds for all loops
3. Avoid heap allocation after init
4. Keep functions single-page length
5. Include min. 2 assertions per function  
6. Minimize data scope
7. Check all non-void return values
8. Use preprocessor only for headers/simple macros
9. Single pointer dereference, no function pointers
10. Address all compiler warnings before release

