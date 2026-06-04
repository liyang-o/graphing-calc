import "./styles.css";

type TokenType = "number" | "identifier" | "operator" | "leftParen" | "rightParen" | "comma" | "eof";

type Token = {
  type: TokenType;
  value: string;
  index: number;
};

type Evaluator = (x: number) => number;

type ExpressionItem = {
  id: number;
  text: string;
  color: string;
  enabled: boolean;
  fn: Evaluator | null;
  error: string | null;
};

type ViewBox = {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
};

const COLORS = ["#c74440", "#2d70b3", "#388c46", "#6042a6", "#fa7e19", "#000000", "#9c27b0", "#00897b"];

const START_EXPRESSIONS = ["sin(x)", "0.25x^2 - 2", "cos(2x)", "sqrt(16 - x^2)"];

const EXAMPLES = [
  "tan(x)",
  "abs(x)",
  "sin(x)/x",
  "log(abs(x))",
  "2sin(x) + cos(3x)",
  "(x+2)(x-3)",
  "sqrt(9-x^2)",
  "1/(x-1)",
];

const FUNCTIONS: Record<string, (...args: number[]) => number> = {
  abs: Math.abs,
  acos: Math.acos,
  acosh: Math.acosh,
  asin: Math.asin,
  asinh: Math.asinh,
  atan: Math.atan,
  atan2: Math.atan2,
  atanh: Math.atanh,
  cbrt: Math.cbrt,
  ceil: Math.ceil,
  cos: Math.cos,
  cosh: Math.cosh,
  exp: Math.exp,
  floor: Math.floor,
  ln: Math.log,
  log: Math.log10,
  log10: Math.log10,
  max: Math.max,
  min: Math.min,
  pow: Math.pow,
  round: Math.round,
  sign: Math.sign,
  sin: Math.sin,
  sinh: Math.sinh,
  sqrt: Math.sqrt,
  tan: Math.tan,
  tanh: Math.tanh,
};

const FUNCTION_ARITY: Record<string, number | "variadic"> = {
  atan2: 2,
  max: "variadic",
  min: "variadic",
  pow: 2,
};

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let index = 0;

  while (index < source.length) {
    const char = source[index];

    if (/\s/.test(char)) {
      index += 1;
      continue;
    }

    if (char === ",") {
      tokens.push({ type: "comma", value: char, index });
      index += 1;
      continue;
    }

    if (char === "(") {
      tokens.push({ type: "leftParen", value: char, index });
      index += 1;
      continue;
    }

    if (char === ")") {
      tokens.push({ type: "rightParen", value: char, index });
      index += 1;
      continue;
    }

    if ("+-*/^".includes(char)) {
      tokens.push({ type: "operator", value: char, index });
      index += 1;
      continue;
    }

    if (/\d|\./.test(char)) {
      const start = index;
      let sawDigit = false;

      while (index < source.length && /\d/.test(source[index])) {
        sawDigit = true;
        index += 1;
      }

      if (source[index] === ".") {
        index += 1;
        while (index < source.length && /\d/.test(source[index])) {
          sawDigit = true;
          index += 1;
        }
      }

      if (!sawDigit) {
        throw new Error(`Unexpected "." at position ${start + 1}`);
      }

      if (source[index]?.toLowerCase() === "e") {
        const exponentStart = index;
        index += 1;
        if (source[index] === "+" || source[index] === "-") {
          index += 1;
        }
        const digitStart = index;
        while (index < source.length && /\d/.test(source[index])) {
          index += 1;
        }
        if (digitStart === index) {
          index = exponentStart;
        }
      }

      tokens.push({ type: "number", value: source.slice(start, index), index: start });
      continue;
    }

    if (/[a-zA-Z_π]/.test(char)) {
      const start = index;
      index += 1;
      while (index < source.length && /[a-zA-Z0-9_]/.test(source[index])) {
        index += 1;
      }
      tokens.push({ type: "identifier", value: source.slice(start, index), index: start });
      continue;
    }

    throw new Error(`Unsupported character "${char}" at position ${index + 1}`);
  }

  tokens.push({ type: "eof", value: "", index: source.length });
  return tokens;
}

class Parser {
  private position = 0;

  constructor(private readonly tokens: Token[]) {}

  parse(): Evaluator {
    const result = this.parseAddSub();
    if (this.current().type !== "eof") {
      throw new Error(`Unexpected token "${this.current().value}"`);
    }
    return result;
  }

  private parseAddSub(): Evaluator {
    let left = this.parseMulDiv();

    while (this.current().type === "operator" && ["+", "-"].includes(this.current().value)) {
      const operator = this.advance().value;
      const right = this.parseMulDiv();
      const previous = left;
      left = operator === "+" ? (x) => previous(x) + right(x) : (x) => previous(x) - right(x);
    }

    return left;
  }

  private parseMulDiv(): Evaluator {
    let left = this.parsePower();

    while (true) {
      if (this.current().type === "operator" && ["*", "/"].includes(this.current().value)) {
        const operator = this.advance().value;
        const right = this.parsePower();
        const previous = left;
        left = operator === "*" ? (x) => previous(x) * right(x) : (x) => previous(x) / right(x);
        continue;
      }

      if (this.startsImplicitMultiplication(this.current())) {
        const right = this.parsePower();
        const previous = left;
        left = (x) => previous(x) * right(x);
        continue;
      }

      return left;
    }
  }

  private parsePower(): Evaluator {
    const left = this.parseUnary();

    if (this.current().type === "operator" && this.current().value === "^") {
      this.advance();
      const right = this.parsePower();
      return (x) => Math.pow(left(x), right(x));
    }

    return left;
  }

  private parseUnary(): Evaluator {
    if (this.current().type === "operator" && this.current().value === "+") {
      this.advance();
      return this.parseUnary();
    }

    if (this.current().type === "operator" && this.current().value === "-") {
      this.advance();
      const value = this.parseUnary();
      return (x) => -value(x);
    }

    return this.parsePrimary();
  }

  private parsePrimary(): Evaluator {
    const token = this.current();

    if (token.type === "number") {
      this.advance();
      const value = Number.parseFloat(token.value);
      return () => value;
    }

    if (token.type === "leftParen") {
      this.advance();
      const expression = this.parseAddSub();
      this.expect("rightParen", "Missing closing parenthesis");
      return expression;
    }

    if (token.type === "identifier") {
      this.advance();
      const name = token.value.toLowerCase();

      if (name === "x") {
        return (x) => x;
      }

      if (name === "pi" || name === "π") {
        return () => Math.PI;
      }

      if (name === "e") {
        return () => Math.E;
      }

      if (name === "tau") {
        return () => Math.PI * 2;
      }

      const fn = FUNCTIONS[name];
      if (!fn) {
        throw new Error(`Unknown identifier "${token.value}"`);
      }

      if (this.current().type === "leftParen") {
        const args = this.parseCallArguments();
        return this.buildFunctionCall(name, fn, args);
      }

      if (this.startsImplicitMultiplication(this.current())) {
        const argument = this.parseUnary();
        return this.buildFunctionCall(name, fn, [argument]);
      }

      throw new Error(`Function "${token.value}" needs an argument, for example ${token.value}(x)`);
    }

    throw new Error(`Expected a number, variable, function, or "("`);
  }

  private parseCallArguments(): Evaluator[] {
    this.expect("leftParen", "Expected opening parenthesis");
    const args: Evaluator[] = [];

    if (this.current().type === "rightParen") {
      this.advance();
      return args;
    }

    while (true) {
      args.push(this.parseAddSub());
      if (this.current().type === "comma") {
        this.advance();
        continue;
      }
      this.expect("rightParen", "Missing closing parenthesis");
      return args;
    }
  }

  private buildFunctionCall(name: string, fn: (...args: number[]) => number, args: Evaluator[]): Evaluator {
    const arity = FUNCTION_ARITY[name] ?? 1;
    if (arity !== "variadic" && args.length !== arity) {
      throw new Error(`Function "${name}" expects ${arity} argument${arity === 1 ? "" : "s"}`);
    }
    if (arity === "variadic" && args.length === 0) {
      throw new Error(`Function "${name}" expects at least one argument`);
    }
    return (x) => fn(...args.map((arg) => arg(x)));
  }

  private startsImplicitMultiplication(token: Token): boolean {
    return token.type === "number" || token.type === "identifier" || token.type === "leftParen";
  }

  private expect(type: TokenType, message: string): Token {
    if (this.current().type !== type) {
      throw new Error(message);
    }
    return this.advance();
  }

  private current(): Token {
    return this.tokens[this.position];
  }

  private advance(): Token {
    const token = this.tokens[this.position];
    this.position += 1;
    return token;
  }
}

function normalizeExpression(source: string): string {
  const trimmed = source.trim();
  const assignment = trimmed.match(/^([a-zA-Z_]\w*\s*\(\s*x\s*\)|y)\s*=/);
  return assignment ? trimmed.slice(assignment[0].length) : trimmed;
}

function compileExpression(source: string): Evaluator {
  const normalized = normalizeExpression(source);
  if (!normalized) {
    throw new Error("Expression is empty");
  }
  return new Parser(tokenize(normalized)).parse();
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "";
  }
  const absolute = Math.abs(value);
  if ((absolute >= 10000 || absolute < 0.001) && absolute !== 0) {
    return value.toExponential(2);
  }
  return Number(value.toPrecision(7)).toString();
}

function niceStep(rawStep: number): number {
  const exponent = Math.floor(Math.log10(rawStep));
  const power = Math.pow(10, exponent);
  const normalized = rawStep / power;

  if (normalized < 1.5) return power;
  if (normalized < 3) return 2 * power;
  if (normalized < 7) return 5 * power;
  return 10 * power;
}

class GraphingCalculator {
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly list: HTMLDivElement;
  private readonly readout: HTMLDivElement;
  private expressions: ExpressionItem[] = [];
  private nextId = 1;
  private view: ViewBox = { xMin: -10, xMax: 10, yMin: -10, yMax: 10 };
  private pointer: { x: number; y: number } | null = null;
  private dragStart: { clientX: number; clientY: number; view: ViewBox } | null = null;

  constructor(root: HTMLElement) {
    root.innerHTML = this.template();

    const canvas = root.querySelector<HTMLCanvasElement>("#graph-canvas");
    const context = canvas?.getContext("2d");
    const list = root.querySelector<HTMLDivElement>("#expression-list");
    const readout = root.querySelector<HTMLDivElement>("#coordinate-readout");

    if (!canvas || !context || !list || !readout) {
      throw new Error("Application template failed to initialize");
    }

    this.canvas = canvas;
    this.context = context;
    this.list = list;
    this.readout = readout;

    START_EXPRESSIONS.forEach((text, index) => this.addExpression(text, COLORS[index % COLORS.length], false));
    this.bindEvents(root);
    this.resizeCanvas();
    this.resetView();
    this.renderExpressionList();
    this.draw();
  }

  private template(): string {
    return `
      <main class="calculator">
        <aside class="sidebar">
          <header class="brand">
            <h1>Graphing Calculator</h1>
            <p>Type expressions in x, for example <strong>sin(x)</strong>, <strong>x^2</strong>, <strong>sqrt(9-x^2)</strong>, or <strong>y = 1/(x-1)</strong>.</p>
          </header>
          <section id="expression-list" class="expression-list" aria-label="Expressions"></section>
          <footer class="sidebar-footer">
            <button id="add-expression" class="add-button" type="button">+ Add expression</button>
            <div class="examples" aria-label="Example expressions">
              ${EXAMPLES.map((example) => `<button type="button" data-example="${example}">${example}</button>`).join("")}
            </div>
          </footer>
        </aside>
        <section class="plot-area">
          <canvas id="graph-canvas" aria-label="Interactive coordinate plane"></canvas>
          <div class="toolbar" aria-label="Graph controls">
            <button id="zoom-in" type="button" title="Zoom in">+</button>
            <button id="zoom-out" type="button" title="Zoom out">−</button>
            <button id="reset-view" type="button" title="Reset view">Reset</button>
            <button id="export-png" type="button" title="Export graph as PNG">PNG</button>
          </div>
          <div id="coordinate-readout" class="coordinate-readout">x: 0, y: 0</div>
          <div class="help">
            Mouse wheel zooms at the cursor, drag pans the plane, and double-click resets. Supported: <code>sin cos tan sqrt abs ln log exp min max pow</code>, constants <code>pi e tau</code>, implicit multiplication like <code>2x</code>.
          </div>
        </section>
      </main>
    `;
  }

  private bindEvents(root: HTMLElement): void {
    root.querySelector<HTMLButtonElement>("#add-expression")?.addEventListener("click", () => {
      this.addExpression("", COLORS[this.expressions.length % COLORS.length]);
    });

    root.querySelectorAll<HTMLButtonElement>("[data-example]").forEach((button) => {
      button.addEventListener("click", () => this.addExpression(button.dataset.example ?? ""));
    });

    root.querySelector<HTMLButtonElement>("#zoom-in")?.addEventListener("click", () => this.zoomAt(0.75));
    root.querySelector<HTMLButtonElement>("#zoom-out")?.addEventListener("click", () => this.zoomAt(1.33));
    root.querySelector<HTMLButtonElement>("#reset-view")?.addEventListener("click", () => {
      this.resetView();
      this.draw();
    });
    root.querySelector<HTMLButtonElement>("#export-png")?.addEventListener("click", () => this.exportPng());

    window.addEventListener("resize", () => {
      this.resizeCanvas();
      this.draw();
    });

    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 0.86 : 1.16;
      this.zoomAt(factor, event.offsetX, event.offsetY);
    }, { passive: false });

    this.canvas.addEventListener("pointerdown", (event) => {
      this.canvas.setPointerCapture(event.pointerId);
      this.canvas.classList.add("dragging");
      this.dragStart = {
        clientX: event.clientX,
        clientY: event.clientY,
        view: { ...this.view },
      };
    });

    this.canvas.addEventListener("pointermove", (event) => {
      this.pointer = { x: event.offsetX, y: event.offsetY };
      this.updateReadout();

      if (!this.dragStart) {
        this.draw();
        return;
      }

      const width = this.canvas.clientWidth;
      const height = this.canvas.clientHeight;
      const dx = ((event.clientX - this.dragStart.clientX) / width) * (this.dragStart.view.xMax - this.dragStart.view.xMin);
      const dy = ((event.clientY - this.dragStart.clientY) / height) * (this.dragStart.view.yMax - this.dragStart.view.yMin);

      this.view = {
        xMin: this.dragStart.view.xMin - dx,
        xMax: this.dragStart.view.xMax - dx,
        yMin: this.dragStart.view.yMin + dy,
        yMax: this.dragStart.view.yMax + dy,
      };
      this.draw();
    });

    this.canvas.addEventListener("pointerleave", () => {
      this.pointer = null;
      this.updateReadout();
      this.draw();
    });

    this.canvas.addEventListener("pointerup", (event) => {
      this.canvas.releasePointerCapture(event.pointerId);
      this.canvas.classList.remove("dragging");
      this.dragStart = null;
    });

    this.canvas.addEventListener("pointercancel", () => {
      this.canvas.classList.remove("dragging");
      this.dragStart = null;
    });

    this.canvas.addEventListener("dblclick", () => {
      this.resetView();
      this.draw();
    });
  }

  private addExpression(text: string, color = COLORS[(this.nextId - 1) % COLORS.length], focus = true): void {
    const expression: ExpressionItem = {
      id: this.nextId,
      text,
      color,
      enabled: true,
      fn: null,
      error: null,
    };
    this.nextId += 1;
    this.compile(expression);
    this.expressions.push(expression);
    this.renderExpressionList();
    this.draw();

    if (focus) {
      requestAnimationFrame(() => {
        const input = this.list.querySelector<HTMLInputElement>(`input[data-id="${expression.id}"]`);
        input?.focus();
        input?.select();
      });
    }
  }

  private renderExpressionList(): void {
    this.list.innerHTML = this.expressions.map((expression) => `
      <article class="expression-card" data-card-id="${expression.id}">
        <input
          class="expression-toggle"
          data-toggle-id="${expression.id}"
          type="checkbox"
          ${expression.enabled ? "checked" : ""}
          style="color: ${expression.color}"
          aria-label="Toggle expression"
        />
        <div>
          <input
            class="expression-input"
            data-id="${expression.id}"
            value="${this.escapeAttribute(expression.text)}"
            placeholder="f(x) ="
            spellcheck="false"
            autocomplete="off"
          />
          <div class="expression-error">${expression.error ?? ""}</div>
        </div>
        <button class="delete-button" data-delete-id="${expression.id}" type="button" aria-label="Delete expression">×</button>
      </article>
    `).join("");

    this.list.querySelectorAll<HTMLInputElement>(".expression-input").forEach((input) => {
      input.addEventListener("input", () => {
        const expression = this.findExpression(input.dataset.id);
        if (!expression) return;
        expression.text = input.value;
        this.compile(expression);
        this.updateExpressionCard(expression);
        this.draw();
      });
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          this.addExpression("");
        }
      });
    });

    this.list.querySelectorAll<HTMLInputElement>(".expression-toggle").forEach((toggle) => {
      toggle.addEventListener("change", () => {
        const expression = this.findExpression(toggle.dataset.toggleId);
        if (!expression) return;
        expression.enabled = toggle.checked;
        this.draw();
      });
    });

    this.list.querySelectorAll<HTMLButtonElement>(".delete-button").forEach((button) => {
      button.addEventListener("click", () => {
        const id = Number(button.dataset.deleteId);
        this.expressions = this.expressions.filter((expression) => expression.id !== id);
        this.renderExpressionList();
        this.draw();
      });
    });
  }

  private updateExpressionCard(expression: ExpressionItem): void {
    const card = this.list.querySelector<HTMLElement>(`[data-card-id="${expression.id}"]`);
    const error = card?.querySelector<HTMLDivElement>(".expression-error");
    if (error) {
      error.textContent = expression.error ?? "";
    }
  }

  private compile(expression: ExpressionItem): void {
    try {
      expression.fn = compileExpression(expression.text);
      expression.error = null;
    } catch (error) {
      expression.fn = null;
      expression.error = error instanceof Error ? error.message : "Invalid expression";
    }
  }

  private findExpression(id: string | undefined): ExpressionItem | undefined {
    return this.expressions.find((expression) => expression.id === Number(id));
  }

  private escapeAttribute(value: string): string {
    return value
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  private resizeCanvas(): void {
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, this.canvas.clientWidth);
    const height = Math.max(1, this.canvas.clientHeight);
    this.canvas.width = Math.floor(width * dpr);
    this.canvas.height = Math.floor(height * dpr);
    this.context.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private resetView(): void {
    const width = Math.max(1, this.canvas.clientWidth);
    const height = Math.max(1, this.canvas.clientHeight);
    const xRange = 20;
    const yRange = xRange * (height / width);
    this.view = {
      xMin: -xRange / 2,
      xMax: xRange / 2,
      yMin: -yRange / 2,
      yMax: yRange / 2,
    };
  }

  private draw(): void {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    this.context.clearRect(0, 0, width, height);
    this.context.fillStyle = "#ffffff";
    this.context.fillRect(0, 0, width, height);
    this.drawGrid(width, height);
    this.drawFunctions(width, height);
    this.drawPointer(width, height);
  }

  private drawGrid(width: number, height: number): void {
    const xRange = this.view.xMax - this.view.xMin;
    const yRange = this.view.yMax - this.view.yMin;
    const xStep = niceStep(xRange / 10);
    const yStep = niceStep(yRange / 10);

    this.context.lineWidth = 1;
    this.context.font = "12px SFMono-Regular, Consolas, monospace";
    this.context.textBaseline = "top";

    this.context.strokeStyle = "#edf0f5";
    this.context.fillStyle = "#748094";
    for (let x = Math.ceil(this.view.xMin / xStep) * xStep; x <= this.view.xMax; x += xStep) {
      const sx = this.worldToScreenX(x, width);
      this.drawLine(sx, 0, sx, height);
      if (Math.abs(x) > xStep / 1000) {
        this.context.fillText(formatNumber(x), sx + 4, this.axisLabelY(height));
      }
    }

    for (let y = Math.ceil(this.view.yMin / yStep) * yStep; y <= this.view.yMax; y += yStep) {
      const sy = this.worldToScreenY(y, height);
      this.drawLine(0, sy, width, sy);
      if (Math.abs(y) > yStep / 1000) {
        this.context.fillText(formatNumber(y), this.axisLabelX(width), sy + 4);
      }
    }

    this.context.strokeStyle = "#5f6b7a";
    this.context.lineWidth = 1.4;
    if (this.view.xMin <= 0 && this.view.xMax >= 0) {
      const xAxis = this.worldToScreenX(0, width);
      this.drawLine(xAxis, 0, xAxis, height);
    }
    if (this.view.yMin <= 0 && this.view.yMax >= 0) {
      const yAxis = this.worldToScreenY(0, height);
      this.drawLine(0, yAxis, width, yAxis);
    }
  }

  private axisLabelX(width: number): number {
    if (this.view.xMin <= 0 && this.view.xMax >= 0) {
      return Math.min(width - 48, Math.max(4, this.worldToScreenX(0, width) + 5));
    }
    return 6;
  }

  private axisLabelY(height: number): number {
    if (this.view.yMin <= 0 && this.view.yMax >= 0) {
      return Math.min(height - 18, Math.max(4, this.worldToScreenY(0, height) + 5));
    }
    return height - 20;
  }

  private drawFunctions(width: number, height: number): void {
    for (const expression of this.expressions) {
      if (!expression.enabled || !expression.fn) {
        continue;
      }

      this.context.save();
      this.context.strokeStyle = expression.color;
      this.context.lineWidth = 2.4;
      this.context.lineJoin = "round";
      this.context.lineCap = "round";
      this.context.beginPath();

      let hasPath = false;
      let previousY: number | null = null;

      for (let sx = 0; sx <= width; sx += 1) {
        const x = this.screenToWorldX(sx, width);
        const y = expression.fn(x);
        const sy = this.worldToScreenY(y, height);

        if (!Number.isFinite(y) || !Number.isFinite(sy) || Math.abs(sy) > height * 100) {
          previousY = null;
          continue;
        }

        const jump = previousY !== null && Math.abs(sy - previousY) > height * 0.8;
        if (!hasPath || previousY === null || jump) {
          this.context.moveTo(sx, sy);
          hasPath = true;
        } else {
          this.context.lineTo(sx, sy);
        }
        previousY = sy;
      }

      this.context.stroke();
      this.context.restore();
    }
  }

  private drawPointer(width: number, height: number): void {
    if (!this.pointer || this.dragStart) {
      return;
    }

    this.context.save();
    this.context.strokeStyle = "rgb(28 115 232 / 24%)";
    this.context.setLineDash([5, 6]);
    this.drawLine(this.pointer.x, 0, this.pointer.x, height);
    this.drawLine(0, this.pointer.y, width, this.pointer.y);
    this.context.restore();
  }

  private drawLine(x1: number, y1: number, x2: number, y2: number): void {
    this.context.beginPath();
    this.context.moveTo(x1, y1);
    this.context.lineTo(x2, y2);
    this.context.stroke();
  }

  private zoomAt(factor: number, screenX?: number, screenY?: number): void {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    const focusX = this.screenToWorldX(screenX ?? width / 2, width);
    const focusY = this.screenToWorldY(screenY ?? height / 2, height);

    this.view = {
      xMin: focusX + (this.view.xMin - focusX) * factor,
      xMax: focusX + (this.view.xMax - focusX) * factor,
      yMin: focusY + (this.view.yMin - focusY) * factor,
      yMax: focusY + (this.view.yMax - focusY) * factor,
    };
    this.draw();
    this.updateReadout();
  }

  private updateReadout(): void {
    if (!this.pointer) {
      this.readout.textContent = "Move over the graph";
      return;
    }
    const x = this.screenToWorldX(this.pointer.x, this.canvas.clientWidth);
    const y = this.screenToWorldY(this.pointer.y, this.canvas.clientHeight);
    this.readout.textContent = `x: ${formatNumber(x)}, y: ${formatNumber(y)}`;
  }

  private exportPng(): void {
    const link = document.createElement("a");
    link.download = "graph.png";
    link.href = this.canvas.toDataURL("image/png");
    link.click();
  }

  private worldToScreenX(x: number, width: number): number {
    return ((x - this.view.xMin) / (this.view.xMax - this.view.xMin)) * width;
  }

  private worldToScreenY(y: number, height: number): number {
    return ((this.view.yMax - y) / (this.view.yMax - this.view.yMin)) * height;
  }

  private screenToWorldX(x: number, width: number): number {
    return this.view.xMin + (x / width) * (this.view.xMax - this.view.xMin);
  }

  private screenToWorldY(y: number, height: number): number {
    return this.view.yMax - (y / height) * (this.view.yMax - this.view.yMin);
  }
}

const root = document.querySelector<HTMLElement>("#app");
if (!root) {
  throw new Error("Missing #app root element");
}

new GraphingCalculator(root);
