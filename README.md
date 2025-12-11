# A latex template
this latex template uses minted for syntax highlighted codeblocks, biber for bibtogrotry and sagetex for CAS and python execution directly in your latex document.

# Requirements
This template requires a uninx like system to run sagetex, linux, mac or wsl, windows is not supported.

install the following:
```
sagemath
biber
```

## Quick Reference Snippets

### Basic SageTex Setup

```latex
\documentclass{article}
\usepackage{sagetex}
\usepackage{pgfplots}
\usepackage{amsmath}

\pgfplotsset{compat=1.18}

\begin{document}
% Your content here
\end{document}
```

### Silent Computation Blocks

Use for calculations that don't need to be displayed:

```latex
\begin{sagesilent}
  x = 5
  y = 10
  result = x * y
\end{sagesilent}
```

### Inline Sage Output

Display computed values inline:

```latex
The result is $\sage{result}$ units.
The rounded value is $\sage{round(pi, 4)}$.
```

### Formatted Output

```latex
The answer is $\sage{latex(result)}$ (formatted as LaTeX)
Numerical: $\sage{result.n(digits=6)}$ (6 decimal places)
```

### Variable Declaration

```latex
\begin{sagesilent}
  var('x y z')  # Declare symbolic variables
  var('t')      # Single variable
\end{sagesilent}
```

### Solving Equations

**Single equation:**
```latex
\begin{sagesilent}
  var('x')
  solution = solve(x^2 - 4 == 0, x)
  x_value = solution[0].rhs()  # Get right-hand side
\end{sagesilent}
```

**System of equations:**
```latex
\begin{sagesilent}
  var('x y')
  sol = solve([x + y == 5, x - y == 1], x, y)
  x_val = sol[0][0].rhs()
  y_val = sol[0][1].rhs()
\end{sagesilent}
```

### Defining Functions

**Simple function:**
```latex
\begin{sagesilent}
  def f(x):
    return x^2 + 2*x + 1
\end{sagesilent}
```

**Recursive function:**
```latex
\begin{sagesilent}
  def fibonacci(n):
    if n <= 1:
      return n
    return fibonacci(n-1) + fibonacci(n-2)
\end{sagesilent}
```

**Function with multiple parameters:**
```latex
\begin{sagesilent}
  def calculate(a, b, operation='add'):
    if operation == 'add':
      return a + b
    elif operation == 'multiply':
      return a * b
    return 0
\end{sagesilent}
```

### Basic Plotting

**Simple function plot:**
```latex
\sageplot{plot(sin(x), (x, 0, 2*pi), color='blue', legend_label='sin(x)')}
```

**Multiple functions:**
```latex
\sageplot{plot(sin(x), (x, 0, 2*pi), color='blue') + plot(cos(x), (x, 0, 2*pi), color='red')}
```

**Data point plotting:**
```latex
\begin{sagesilent}
  data = [(0, 1), (1, 2), (2, 4), (3, 8)]
\end{sagesilent}

\sageplot{list_plot(data, plotjoined=True, marker='o', color='blue')}
```

**Plot with customization:**
```latex
\sageplot{plot(x^2, (x, -5, 5), 
  color='red', 
  thickness=2, 
  legend_label='$f(x) = x^2$',
  axes_labels=['$x$', '$y$'])}
```

### Generating Data

**List comprehension:**
```latex
\begin{sagesilent}
  points = [(i, i^2) for i in range(10)]
\end{sagesilent}
```

**Using loops:**
```latex
\begin{sagesilent}
  data_points = []
  for i in range(100):
    x = i * 0.1
    y = sin(x)
    data_points.append((x, y))
\end{sagesilent}
```

**Function-generated data:**
```latex
\begin{sagesilent}
  def generate_data(func, start, end, steps):
    h = (end - start) / steps
    return [(start + i*h, func(start + i*h)) for i in range(steps+1)]
  
  my_data = generate_data(lambda x: x^2, 0, 10, 100)
\end{sagesilent}
```

### Calculus Operations

**Derivatives:**
```latex
\begin{sagesilent}
  var('x')
  f = x^3 + 2*x^2 - x + 1
  f_prime = derivative(f, x)
  f_double_prime = derivative(f, x, 2)
\end{sagesilent}
```

**Integration:**
```latex
\begin{sagesilent}
  var('x')
  # Indefinite integral
  antiderivative = integrate(x^2, x)
  
  # Definite integral
  area = integrate(x^2, x, 0, 5)
\end{sagesilent}
```

**Limits:**
```latex
\begin{sagesilent}
  var('x')
  result = limit((sin(x))/x, x=0)
\end{sagesilent}
```

### Matrix Operations

```latex
\begin{sagesilent}
  A = matrix([[1, 2], [3, 4]])
  B = matrix([[5, 6], [7, 8]])
  
  # Operations
  C = A * B           # Matrix multiplication
  A_inv = A.inverse() # Inverse
  det_A = A.det()     # Determinant
  tr_A = A.trace()    # Trace
\end{sagesilent}
```

### Linear Algebra

```latex
\begin{sagesilent}
  A = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  
  # Eigenvalues and eigenvectors
  eigenvals = A.eigenvalues()
  eigenvects = A.eigenvectors_right()
  
  # Rank
  rank = A.rank()
\end{sagesilent}
```

### Polynomial Operations

```latex
\begin{sagesilent}
  var('x')
  p = x^3 + 2*x^2 - x + 1
  
  # Roots
  roots = p.roots()
  
  # Factorization
  factored = factor(p)
  
  # Expansion
  expanded = expand((x+1)*(x+2)*(x+3))
\end{sagesilent}
```

### Statistics & Data Analysis

```latex
\begin{sagesilent}
  data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  
  mean_val = mean(data)
  median_val = median(data)
  std_dev = std(data)
  variance = variance(data)
\end{sagesilent}
```

### Using NumPy

```latex
\begin{sagesilent}
  import numpy as np
  
  # Arrays
  arr = np.array([1, 2, 3, 4, 5])
  
  # Operations
  arr_squared = arr ** 2
  arr_sum = np.sum(arr)
  arr_mean = np.mean(arr)
\end{sagesilent}
```

### Curve Fitting

**Polynomial fitting:**
```latex
\begin{sagesilent}
  import numpy as np
  
  # Data points
  x_data = [1, 2, 3, 4, 5]
  y_data = [2, 4, 5, 4, 5]
  
  # Fit polynomial of degree 2
  coeffs = np.polyfit(x_data, y_data, 2)
  poly = np.poly1d(coeffs)
  
  # Calculate R²
  y_pred = poly(x_data)
  r_squared = np.corrcoef(y_data, y_pred)[0, 1]**2
\end{sagesilent}
```

**Convert to Sage polynomial:**
```latex
\begin{sagesilent}
  var('x')
  degree = 2
  poly_sage = sum(coeffs[i]*x^(degree-i) for i in range(degree+1))
\end{sagesilent}
```

### Advanced Plotting

**Parametric plots:**
```latex
\sageplot{parametric_plot((cos(t), sin(t)), (t, 0, 2*pi), color='purple')}
```

**3D plots:**
```latex
\begin{sagesilent}
  var('x y')
  f(x, y) = x^2 + y^2
\end{sagesilent}

\sageplot{plot3d(f, (x, -2, 2), (y, -2, 2))}
```

**Contour plots:**
```latex
\sageplot{contour_plot(x^2 + y^2, (x, -3, 3), (y, -3, 3), fill=False)}
```

**Multiple plots combined:**
```latex
\begin{sagesilent}
  p1 = plot(sin(x), (x, 0, 2*pi), color='blue')
  p2 = plot(cos(x), (x, 0, 2*pi), color='red')
  p3 = list_plot([(0, 0), (pi, 0), (2*pi, 0)], size=50, color='green')
  combined = p1 + p2 + p3
\end{sagesilent}

\sageplot{combined}
```

### Numerical Methods

**Newton's method:**
```latex
\begin{sagesilent}
  def newtons_method(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
      fx = f(x)
      if abs(fx) < tol:
        return x
      x = x - fx/df(x)
    return x
\end{sagesilent}
```

**Euler's method for ODEs:**
```latex
\begin{sagesilent}
  def euler_method(f, t0, y0, t_end, h):
    t_vals = [t0]
    y_vals = [y0]
    t = t0
    y = y0
    
    while t < t_end:
      y = y + h * f(t, y)
      t = t + h
      t_vals.append(t)
      y_vals.append(y)
    
    return list(zip(t_vals, y_vals))
\end{sagesilent}
```

### Tables from Data

```latex
\begin{sagesilent}
  data = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
  
  table_str = r"\begin{tabular}{|c|c|c|}" + "\n"
  table_str += r"\hline" + "\n"
  for row in data:
    table_str += " & ".join(str(x) for x in row) + r" \\" + "\n"
    table_str += r"\hline" + "\n"
  table_str += r"\end{tabular}"
\end{sagesilent}

\sage{table_str}
```

---

## Techniques Explained

### 1. SageTex Integration Basics

SageTex bridges LaTeX and Sage (Python-based mathematics software), enabling:

- **Computation**: Perform calculations directly in your document
- **Symbolic mathematics**: Solve equations, differentiate, integrate
- **Data visualization**: Generate plots from computed data
- **Dynamic content**: Values update automatically when parameters change

**Three main constructs:**

- `\begin{sagesilent}...\end{sagesilent}` - Execute code without output
- `\sage{expression}` - Display inline results
- `\sageplot{plot_command}` - Embed plots

### 2. When to Use SageTex

**Perfect for:**
- Homework assignments with calculations
- Research papers with computational results
- Technical reports requiring plots
- Documents where values change frequently
- Teaching materials with examples

**Not ideal for:**
- Simple static documents
- Documents requiring portability (SageTex needs special compilation)
- When collaborators don't have Sage installed

### 3. Silent Blocks vs Display

**Silent blocks** (`sagesilent`) are for:
- Variable declarations
- Function definitions
- Data processing
- Complex calculations
- Anything you don't want printed

**Display commands** (`\sage{}`) are for:
- Showing final results
- Inserting computed values
- Displaying formatted equations

### 4. Working with Functions

Functions in SageTex work like Python:

```python
def function_name(parameters):
  # code
  return result
```

**Use cases:**
- Recursive sequences
- Custom calculations
- Data generation
- Numerical methods
- Repeated operations

### 5. Plotting Strategies

**For continuous functions:**
```latex
plot(function, (variable, start, end))
```

**For discrete data:**
```latex
list_plot(points, plotjoined=True)
```

**For comparisons:**
- Plot multiple functions on same axes
- Use different colors and line styles
- Add legend labels

### 6. Data Generation Patterns

**Pattern 1: Direct list comprehension**
```python
data = [(x, f(x)) for x in range(start, end)]
```

**Pattern 2: Loop with accumulation**
```python
data = []
for condition:
  data.append((x, y))
```

**Pattern 3: Function generator**
```python
def generate(func, params):
  return [computation]
```

### 7. Curve Fitting Workflow

1. **Generate or collect data points**
2. **Choose appropriate model** (polynomial, exponential, etc.)
3. **Fit the model** using NumPy or Sage
4. **Evaluate fit quality** (R², visual inspection)
5. **Use fitted model** for predictions or analysis

### 8. Numerical Methods

Common patterns for implementing numerical methods:

**Iterative methods:**
- Initialize starting value
- Loop until convergence or max iterations
- Update value each iteration
- Return final result

**Differential equations:**
- Define step size
- Initialize conditions
- Step forward (or backward) in time
- Store trajectory

### 9. Integration with LaTeX Math

SageTex results can be embedded in any LaTeX math environment:

```latex
\begin{align*}
  f(x) &= \sage{f} \\
  f'(x) &= \sage{derivative(f, x)} \\
  \int f(x)\,dx &= \sage{integrate(f, x)}
\end{align*}
```

### 10. Error Handling

**Common issues:**

- **Undefined variables**: Declare with `var('x')`
- **Import errors**: Use `import` statements in `sagesilent` blocks
- **Precision issues**: Use `.n(digits=N)` for numerical approximation
- **Plot errors**: Check variable ranges and function domains

### 11. Optimization Tips

**For large computations:**
- Cache results in variables
- Avoid recomputing in multiple `\sage{}` calls
- Use NumPy for array operations
- Simplify symbolic expressions when possible

**For many plots:**
- Combine plots before calling `\sageplot`
- Reuse plot objects
- Adjust plot resolution if needed

### 12. Document Organization

**Best practices:**

1. **Setup section**: All imports and configurations at top
2. **Function definitions**: Define reusable functions early
3. **Section-specific code**: Computation blocks near where results are used
4. **Modular approach**: Break complex computations into functions

**Example structure:**
```latex
% Preamble and packages

\begin{document}

% Global setup
\begin{sagesilent}
  # Imports
  # Global variables
  # Utility functions
\end{sagesilent}

\section{Section 1}
\begin{sagesilent}
  # Section-specific computations
\end{sagesilent}
% Content with \sage{} and \sageplot{}

\end{document}
```

### 13. Combining Sage and LaTeX Formatting

**Format numbers:**
```latex
$\sage{round(value, 2)}$ (rounded)
$\sage{value.n(digits=4)}$ (numerical)
```

**Format expressions:**
```latex
$\sage{latex(expression)}$ (as LaTeX)
```

**Custom formatting:**
```python
formatted = f"${value:.2f}$"  # Python f-string
```

### 14. Advanced Topics

**Symbolic manipulation:**
- Simplification: `simplify(expr)`
- Expansion: `expand(expr)`
- Factorization: `factor(expr)`
- Substitution: `expr.subs(x=value)`

**Differential equations:**
```python
desolve(de, dependent_var, ics=[initial_conditions])
```

**Vector calculus:**
```python
var('x y z')
F = vector([x*y, y*z, z*x])
div_F = F.div()
curl_F = F.curl()
```

---

## Compilation Process

SageTex requires a three-step compilation:

```bash
pdflatex document.tex    # Creates .sage file
sage document.sagetex.sage    # Runs Sage computations
pdflatex document.tex    # Incorporates results
```

**For table of contents, run pdflatex twice at the end:**
```bash
pdflatex document.tex
sage document.sagetex.sage
pdflatex document.tex
pdflatex document.tex
```

**Makefile example:**
```makefile
document.pdf: document.tex
  pdflatex document.tex
  sage document.sagetex.sage
  pdflatex document.tex
  pdflatex document.tex
```

---

## Useful Packages

**Essential:**
- `sagetex` - Core integration
- `amsmath` - Advanced math typesetting
- `graphicx` - Image handling

**Recommended:**
- `pgfplots` - Advanced plotting
- `tikz` - Graphics and diagrams
- `geometry` - Page layout
- `booktabs` - Professional tables
- `hyperref` - Clickable links

**Setup example:**
```latex
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{pgfplots}
\usepackage{sagetex}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{hyperref}

\pgfplotsset{compat=1.18}
\geometry{margin=1in}
```

---

## Troubleshooting

**"Undefined control sequence" errors:**
- Check all `\sage{}` commands have valid syntax
- Ensure variables are defined in `sagesilent` blocks

**Plots not appearing:**
- Verify `\sageplot{}` syntax
- Check that plot commands return plot objects
- Ensure variable ranges are valid

**Compilation hangs:**
- Check for infinite loops in recursive functions
- Verify numerical methods have convergence criteria
- Look for division by zero or undefined operations

**"File not found" errors:**
- Ensure `.sagetex.sage` file was generated
- Run all compilation steps in order
- Check file permissions

---

## Best Practices

1. **Comment your code** - Explain complex calculations
2. **Use meaningful variable names** - Makes debugging easier
3. **Test incrementally** - Compile frequently during development
4. **Separate concerns** - Functions for computation, separate blocks for plotting
5. **Handle edge cases** - Check for division by zero, invalid ranges
6. **Optimize when needed** - Profile slow computations
7. **Version control** - Track both `.tex` and generated files
8. **Document dependencies** - Note required packages and versions

---

## Resources

- **Sage Documentation**: https://doc.sagemath.org/
- **SageTex Documentation**: https://doc.sagemath.org/html/en/tutorial/sagetex.html
- **LaTeX Documentation**: https://www.latex-project.org/help/documentation/
- **CTAN (LaTeX packages)**: https://www.ctan.org/

---

## Example Workflow

1. **Plan your document** - What computations do you need?
2. **Set up skeleton** - Sections, basic structure
3. **Add computations** - Start with simple cases
4. **Test frequently** - Compile and check output
5. **Add visualizations** - Plots and figures
6. **Refine and format** - Polish presentation
7. **Final compilation** - Clean build from scratch

This workflow ensures you catch errors early and build confidence as you go!
