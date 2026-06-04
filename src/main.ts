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

type HoverValue = {
  expression: ExpressionItem;
  y: number;
  screenY: number;
  inView: boolean;
};

type SidebarTab = "expressions" | "extractor";

type ExtractedFormula = {
  formula: string;
  plotExpression: string | null;
  note: string;
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

const CODE_PLACEHOLDER = `sum(square(command_vel[:, :2] - robot_vel_yaw[:, :2]), dim=1)
robot_vel_b = _body_vectors_in_anchor_frame(robot_rel_vel_w, command.robot_anchor_quat_w)
diff = ref_vel_b - robot_vel_b
return torch.exp(-(diff * diff).sum(dim=-1).mean(dim=-1) / (std * std))`;

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
  private readonly root: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly list: HTMLDivElement;
  private readonly readout: HTMLDivElement;
  private readonly codeInput: HTMLTextAreaElement;
  private readonly extractedList: HTMLDivElement;
  private expressions: ExpressionItem[] = [];
  private nextId = 1;
  private gridDivisions = 10;
  private view: ViewBox = { xMin: -10, xMax: 10, yMin: -10, yMax: 10 };
  private pointer: { x: number; y: number } | null = null;
  private dragStart: { clientX: number; clientY: number; view: ViewBox } | null = null;

  constructor(root: HTMLElement) {
    root.innerHTML = this.template();

    const canvas = root.querySelector<HTMLCanvasElement>("#graph-canvas");
    const context = canvas?.getContext("2d");
    const list = root.querySelector<HTMLDivElement>("#expression-list");
    const readout = root.querySelector<HTMLDivElement>("#coordinate-readout");
    const codeInput = root.querySelector<HTMLTextAreaElement>("#code-input");
    const extractedList = root.querySelector<HTMLDivElement>("#extracted-formulas");

    if (!canvas || !context || !list || !readout || !codeInput || !extractedList) {
      throw new Error("Application template failed to initialize");
    }

    this.root = root;
    this.canvas = canvas;
    this.context = context;
    this.list = list;
    this.readout = readout;
    this.codeInput = codeInput;
    this.extractedList = extractedList;

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
          <nav class="sidebar-tabs" aria-label="Sidebar tools">
            <button class="sidebar-tab active" type="button" data-sidebar-tab="expressions" aria-selected="true">Expressions</button>
            <button class="sidebar-tab" type="button" data-sidebar-tab="extractor" aria-selected="false">Extract formulas</button>
          </nav>
          <section id="expressions-panel" class="sidebar-panel active" aria-label="Expression input panel">
            <section id="expression-list" class="expression-list" aria-label="Expressions"></section>
            <footer class="sidebar-footer">
              <button id="add-expression" class="add-button" type="button">+ Add expression</button>
              <div class="examples" aria-label="Example expressions">
                ${EXAMPLES.map((example) => `<button type="button" data-example="${example}">${example}</button>`).join("")}
              </div>
            </footer>
          </section>
          <section id="extractor-panel" class="sidebar-panel" aria-label="Code formula extraction panel" hidden>
            <div class="extractor">
              <p class="extractor-help">Paste Python or C++ code. The extractor skips control lines and keeps assignment/return formulas that look mathematical.</p>
              <textarea id="code-input" class="code-input" spellcheck="false" placeholder="${CODE_PLACEHOLDER}"></textarea>
              <div class="extractor-actions">
                <button id="extract-formulas" class="add-button" type="button">Extract formulas</button>
                <button id="clear-code" class="secondary-button" type="button">Clear</button>
              </div>
              <section class="extracted-results" aria-label="Extracted formulas">
                <div class="result-header">
                  <strong>Extracted formulas</strong>
                  <button id="copy-all-formulas" class="secondary-button compact" type="button">Copy all</button>
                </div>
                <div id="extracted-formulas" class="extracted-formulas">
                  <p class="empty-result">Extracted formulas will appear here.</p>
                </div>
              </section>
            </div>
          </section>
        </aside>
        <section class="plot-area">
          <canvas id="graph-canvas" aria-label="Interactive coordinate plane"></canvas>
          <div class="toolbar" aria-label="Graph controls">
            <button id="zoom-in" type="button" title="Zoom in">+</button>
            <button id="zoom-out" type="button" title="Zoom out">−</button>
            <button id="reset-view" type="button" title="Reset view">Reset</button>
            <button id="export-png" type="button" title="Export graph as PNG">PNG</button>
            <label class="grid-control" title="Adjust grid size">
              <span>Grid</span>
              <input id="grid-size" type="range" min="4" max="24" step="1" value="${this.gridDivisions}" />
              <span id="grid-size-value">${this.gridDivisions}</span>
            </label>
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
    root.querySelectorAll<HTMLButtonElement>("[data-sidebar-tab]").forEach((button) => {
      button.addEventListener("click", () => this.switchSidebarTab(button.dataset.sidebarTab as SidebarTab));
    });

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
      this.updateReadout();
    });
    root.querySelector<HTMLButtonElement>("#export-png")?.addEventListener("click", () => this.exportPng());
    root.querySelector<HTMLInputElement>("#grid-size")?.addEventListener("input", (event) => {
      const slider = event.currentTarget as HTMLInputElement;
      this.gridDivisions = Number(slider.value);
      const value = root.querySelector<HTMLSpanElement>("#grid-size-value");
      if (value) {
        value.textContent = String(this.gridDivisions);
      }
      this.draw();
      this.updateReadout();
    });

    root.querySelector<HTMLButtonElement>("#extract-formulas")?.addEventListener("click", () => {
      this.renderExtractedFormulas(this.extractFormulasFromCode(this.codeInput.value));
    });
    root.querySelector<HTMLButtonElement>("#clear-code")?.addEventListener("click", () => {
      this.codeInput.value = "";
      this.renderExtractedFormulas([]);
      this.codeInput.focus();
    });
    root.querySelector<HTMLButtonElement>("#copy-all-formulas")?.addEventListener("click", (event) => {
      const formulas = this.getRenderedFormulas();
      if (formulas.length === 0) {
        return;
      }
      void this.copyText(formulas.join("\n"), event.currentTarget as HTMLButtonElement);
    });

    window.addEventListener("resize", () => {
      this.resizeCanvas();
      this.draw();
      this.updateReadout();
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

      if (!this.dragStart) {
        this.updateReadout();
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
      this.updateReadout();
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
      this.updateReadout();
      this.draw();
    });

    this.canvas.addEventListener("pointercancel", () => {
      this.canvas.classList.remove("dragging");
      this.dragStart = null;
    });

    this.canvas.addEventListener("dblclick", () => {
      this.resetView();
      this.draw();
      this.updateReadout();
    });
  }

  private switchSidebarTab(tab: SidebarTab): void {
    if (tab !== "expressions" && tab !== "extractor") {
      return;
    }

    this.root.querySelectorAll<HTMLButtonElement>("[data-sidebar-tab]").forEach((button) => {
      const isActive = button.dataset.sidebarTab === tab;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-selected", String(isActive));
    });

    const expressionsPanel = this.root.querySelector<HTMLElement>("#expressions-panel");
    const extractorPanel = this.root.querySelector<HTMLElement>("#extractor-panel");

    if (expressionsPanel && extractorPanel) {
      expressionsPanel.hidden = tab !== "expressions";
      extractorPanel.hidden = tab !== "extractor";
      expressionsPanel.classList.toggle("active", tab === "expressions");
      extractorPanel.classList.toggle("active", tab === "extractor");
    }

    if (tab === "extractor") {
      this.codeInput.focus();
    }
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
    this.updateReadout();

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
        this.updateReadout();
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
        this.updateReadout();
      });
    });

    this.list.querySelectorAll<HTMLButtonElement>(".delete-button").forEach((button) => {
      button.addEventListener("click", () => {
        const id = Number(button.dataset.deleteId);
        this.expressions = this.expressions.filter((expression) => expression.id !== id);
        this.renderExpressionList();
        this.draw();
        this.updateReadout();
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
    const xStep = niceStep(xRange / this.gridDivisions);
    const yStep = niceStep(yRange / this.gridDivisions);

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

    const x = this.screenToWorldX(this.pointer.x, width);
    const hoverValues = this.getHoverValues(x, height);

    this.context.save();
    this.context.strokeStyle = "rgb(28 115 232 / 24%)";
    this.context.setLineDash([5, 6]);
    this.drawLine(this.pointer.x, 0, this.pointer.x, height);
    this.context.restore();

    for (const value of hoverValues) {
      if (!value.inView) {
        continue;
      }

      this.context.save();
      this.context.beginPath();
      this.context.arc(this.pointer.x, value.screenY, 4.5, 0, Math.PI * 2);
      this.context.fillStyle = "#fff";
      this.context.fill();
      this.context.lineWidth = 2.5;
      this.context.strokeStyle = value.expression.color;
      this.context.stroke();
      this.context.restore();
    }
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
    const hoverValues = this.getHoverValues(x, this.canvas.clientHeight);
    const header = document.createElement("div");
    header.className = "readout-header";
    header.textContent = `x: ${formatNumber(x)}`;

    if (hoverValues.length === 0) {
      const empty = document.createElement("div");
      empty.className = "readout-empty";
      empty.textContent = "No active function values";
      this.readout.replaceChildren(header, empty);
      return;
    }

    const list = document.createElement("div");
    list.className = "readout-series";

    for (const value of hoverValues) {
      const row = document.createElement("div");
      row.className = "readout-row";

      const marker = document.createElement("span");
      marker.className = "readout-marker";
      marker.style.backgroundColor = value.expression.color;

      const label = document.createElement("span");
      label.className = "readout-label";
      label.textContent = value.expression.text.trim() || `Expression ${value.expression.id}`;

      const yValue = document.createElement("span");
      yValue.className = "readout-value";
      yValue.textContent = formatNumber(value.y);

      row.replaceChildren(marker, label, yValue);
      list.append(row);
    }

    this.readout.replaceChildren(header, list);
  }

  private getHoverValues(x: number, height: number): HoverValue[] {
    return this.expressions.flatMap((expression) => {
      if (!expression.enabled || !expression.fn) {
        return [];
      }

      const y = expression.fn(x);
      const screenY = this.worldToScreenY(y, height);

      if (!Number.isFinite(y) || !Number.isFinite(screenY)) {
        return [];
      }

      return [{
        expression,
        y,
        screenY,
        inView: screenY >= 0 && screenY <= height,
      }];
    });
  }

  private extractFormulasFromCode(source: string): ExtractedFormula[] {
    const formulas: ExtractedFormula[] = [];
    const seen = new Set<string>();

    for (const line of this.getLogicalCodeLines(source)) {
      if (this.isNonFormulaCodeLine(line)) {
        continue;
      }

      const extracted = this.extractFormulaFromLine(line);
      if (!extracted || !this.looksLikeFormula(extracted.rawFormula)) {
        continue;
      }

      const symbolicExpression = this.toSymbolicFormula(extracted.rawExpression);
      const formula = extracted.target ? `${extracted.target} = ${symbolicExpression}` : symbolicExpression;
      if (seen.has(formula)) {
        continue;
      }

      const plotExpression = this.toPlotExpression(extracted.rawExpression);
      formulas.push({
        formula,
        plotExpression,
        note: plotExpression
          ? "Can be added to the graph as an x expression."
          : "Formula result only. Tensor, vector, matrix, or non-x variables are not plotted by this graph.",
      });
      seen.add(formula);
    }

    return formulas;
  }

  private getLogicalCodeLines(source: string): string[] {
    const lines: string[] = [];
    let buffer = "";
    let depth = 0;

    for (const rawLine of source.split(/\r?\n/)) {
      const line = this.stripCodeComments(rawLine);
      if (!line) {
        continue;
      }

      buffer = buffer ? `${buffer} ${line}` : line;
      depth += this.countMatches(line, /[\[{(]/g) - this.countMatches(line, /[\]})]/g);

      const continues = line.endsWith("\\") || /[,+\-*\/(]$/.test(line);
      if (depth <= 0 && !continues) {
        lines.push(buffer.replace(/\\\s*/g, " ").trim());
        buffer = "";
        depth = 0;
      }
    }

    if (buffer) {
      lines.push(buffer.trim());
    }

    return lines;
  }

  private stripCodeComments(line: string): string {
    return line
      .replace(/\/\/.*$/, "")
      .replace(/#.*$/, "")
      .trim();
  }

  private countMatches(source: string, pattern: RegExp): number {
    return source.match(pattern)?.length ?? 0;
  }

  private isNonFormulaCodeLine(line: string): boolean {
    const compact = line.trim();
    return /^(?:import|from|using|namespace|class|struct|enum|def|template|public:|private:|protected:)\b/.test(compact)
      || /^(?:if|else|elif|for|while|switch|case|try|catch|with)\b/.test(compact)
      || /^[{}()[\];,]*$/.test(compact)
      || /^#/.test(compact);
  }

  private extractFormulaFromLine(line: string): { target: string | null; rawExpression: string; rawFormula: string } | null {
    const cleaned = line.trim().replace(/[;{}]+$/g, "").trim();
    const returnMatch = cleaned.match(/^return\s+(.+)$/);
    if (returnMatch) {
      const rawExpression = returnMatch[1].trim();
      return {
        target: null,
        rawExpression,
        rawFormula: rawExpression,
      };
    }

    const assignmentMatch = cleaned.match(/^(.+?)\s*(\+=|-=|\*=|\/=|%=|=(?!=))\s*(.+)$/);
    if (assignmentMatch) {
      const [, rawTarget, operator, rawExpression] = assignmentMatch;
      const target = this.normalizeAssignmentTarget(rawTarget);
      const expression = rawExpression.trim();
      if (!target || !expression) {
        return null;
      }

      if (operator === "=") {
        return {
          target,
          rawExpression: expression,
          rawFormula: `${target} = ${expression}`,
        };
      }

      return {
        target,
        rawExpression: `${target} ${operator[0]} (${expression})`,
        rawFormula: `${target} = ${target} ${operator[0]} (${expression})`,
      };
    }

    const expression = cleaned.trim();
    if (expression && this.looksLikeFormula(expression)) {
      return {
        target: null,
        rawExpression: expression,
        rawFormula: expression,
      };
    }

    return null;
  }

  private normalizeAssignmentTarget(source: string): string {
    const withoutQualifiers = source
      .replace(/\b(?:const|constexpr|static|volatile|mutable|inline)\b/g, "")
      .trim();
    const parts = withoutQualifiers.split(/\s+/);
    return (parts[parts.length - 1] ?? "")
      .replace(/^[*&]+/, "")
      .replace(/\[[^\]]*]$/g, "")
      .trim();
  }

  private normalizeCodeFormula(source: string): string {
    return source
      .trim()
      .replace(/[;{}]+$/g, "")
      .replace(/\b(?:torch|math|np|numpy)\./g, "")
      .replace(/\bstd::/g, "")
      .replace(/->/g, ".")
      .replace(/\*\*/g, "^")
      .replace(/\s+/g, " ");
  }

  private toSymbolicFormula(source: string): string {
    let formula = this.normalizeCodeFormula(source);
    formula = this.rewriteFunctionCalls(formula, "square", (args) => `(${args[0] ?? ""})^2`);
    formula = this.rewriteFunctionCalls(formula, "pow", (args) => `(${args[0] ?? ""})^(${args[1] ?? ""})`);
    formula = this.rewriteFunctionCalls(formula, "sum", (args) => this.formatReduction("Σ", args));
    formula = this.rewriteFunctionCalls(formula, "mean", (args) => this.formatReduction("mean", args));
    formula = this.rewriteMethodReductions(formula);
    return formula
      .replace(/\s*\*\s*/g, " · ")
      .replace(/\s*\/\s*/g, " / ")
      .replace(/\s+-\s+/g, " - ")
      .replace(/\s*\+\s*/g, " + ")
      .replace(/\s+/g, " ")
      .trim();
  }

  private toPlotExpression(source: string): string | null {
    let expression = this.normalizeCodeFormula(source);
    expression = this.rewriteFunctionCalls(expression, "square", (args) => `(${args[0] ?? ""})^2`);

    if (!this.isPlotCompatibleExpression(expression)) {
      return null;
    }

    try {
      compileExpression(expression);
      return expression;
    } catch {
      return null;
    }
  }

  private rewriteFunctionCalls(
    source: string,
    name: string,
    replacer: (args: string[]) => string,
  ): string {
    let result = source;
    let searchFrom = 0;

    while (searchFrom < result.length) {
      const match = this.findFunctionCall(result, name, searchFrom);
      if (!match) {
        break;
      }

      const args = this.splitTopLevelArgs(result.slice(match.argsStart, match.argsEnd));
      const replacement = replacer(args);
      result = `${result.slice(0, match.start)}${replacement}${result.slice(match.end + 1)}`;
      searchFrom = match.start + replacement.length;
    }

    return result;
  }

  private findFunctionCall(
    source: string,
    name: string,
    searchFrom: number,
  ): { start: number; argsStart: number; argsEnd: number; end: number } | null {
    const pattern = `${name}(`;
    let start = source.indexOf(pattern, searchFrom);

    while (start !== -1) {
      const previous = source[start - 1] ?? "";
      if (!/[a-zA-Z0-9_.]/.test(previous)) {
        const openParen = start + name.length;
        const closeParen = this.findMatchingParen(source, openParen);
        if (closeParen !== -1) {
          return {
            start,
            argsStart: openParen + 1,
            argsEnd: closeParen,
            end: closeParen,
          };
        }
      }
      start = source.indexOf(pattern, start + pattern.length);
    }

    return null;
  }

  private splitTopLevelArgs(source: string): string[] {
    const args: string[] = [];
    let depth = 0;
    let start = 0;

    for (let index = 0; index < source.length; index += 1) {
      const char = source[index];
      if ("([{".includes(char)) {
        depth += 1;
      } else if (")]}".includes(char)) {
        depth -= 1;
      } else if (char === "," && depth === 0) {
        args.push(source.slice(start, index).trim());
        start = index + 1;
      }
    }

    const last = source.slice(start).trim();
    if (last) {
      args.push(last);
    }

    return args;
  }

  private findMatchingParen(source: string, openParen: number): number {
    let depth = 0;
    for (let index = openParen; index < source.length; index += 1) {
      const char = source[index];
      if (char === "(") {
        depth += 1;
      } else if (char === ")") {
        depth -= 1;
        if (depth === 0) {
          return index;
        }
      }
    }
    return -1;
  }

  private formatReduction(symbol: "Σ" | "mean", args: string[]): string {
    const value = args[0] ?? "";
    const dims = args.slice(1).filter((arg) => /\bdim\s*=/.test(arg));
    const suffix = dims.length > 0 ? `_{${dims.join(", ")}}` : "";
    return `${symbol}${suffix}(${value})`;
  }

  private rewriteMethodReductions(source: string): string {
    let result = source;
    let changed = true;

    while (changed) {
      changed = false;
      for (const method of ["sum", "mean"] as const) {
        const marker = `.${method}(`;
        const markerIndex = result.indexOf(marker);
        if (markerIndex === -1) {
          continue;
        }

        const argsOpen = markerIndex + marker.length - 1;
        const argsEnd = this.findMatchingParen(result, argsOpen);
        if (argsEnd === -1) {
          continue;
        }

        const baseStart = this.findMethodBaseStart(result, markerIndex - 1);
        const base = result.slice(baseStart, markerIndex).trim();
        const args = this.splitTopLevelArgs(result.slice(argsOpen + 1, argsEnd));
        const replacement = this.formatReduction(method === "sum" ? "Σ" : "mean", [base, ...args]);
        result = `${result.slice(0, baseStart)}${replacement}${result.slice(argsEnd + 1)}`;
        changed = true;
        break;
      }
    }

    return result;
  }

  private findMethodBaseStart(source: string, baseEnd: number): number {
    let index = baseEnd;
    while (index >= 0 && /\s/.test(source[index])) {
      index -= 1;
    }

    if (source[index] === ")") {
      const openParen = this.findMatchingOpenParen(source, index);
      if (openParen !== -1) {
        index = openParen - 1;
        while (index >= 0 && /[a-zA-Z0-9_Σ{}=:\-]/.test(source[index])) {
          index -= 1;
        }
        return index + 1;
      }
    }

    while (index >= 0 && !/[\s+\-*\/%^=,<>&|!?;]/.test(source[index])) {
      index -= 1;
    }
    return index + 1;
  }

  private findMatchingOpenParen(source: string, closeParen: number): number {
    let depth = 0;
    for (let index = closeParen; index >= 0; index -= 1) {
      const char = source[index];
      if (char === ")") {
        depth += 1;
      } else if (char === "(") {
        depth -= 1;
        if (depth === 0) {
          return index;
        }
      }
    }
    return -1;
  }

  private isPlotCompatibleExpression(expression: string): boolean {
    if (/[\[\]:]|\.|Σ|_{|->/.test(expression)) {
      return false;
    }

    const names = expression.match(/[a-zA-Z_]\w*/g) ?? [];
    return names.every((name) => {
      const normalized = name.toLowerCase();
      return normalized === "x"
        || normalized === "pi"
        || normalized === "e"
        || normalized === "tau"
        || normalized in FUNCTIONS;
    });
  }

  private looksLikeFormula(formula: string): boolean {
    const assignment = formula.match(/^[a-zA-Z_][\w.\[\]]*\s*=\s*(.+)$/);
    const expression = assignment?.[1] ?? formula;
    return /[+\-*\/^%]/.test(expression)
      || /\b(?:sin|cos|tan|asin|acos|atan|sqrt|abs|exp|log|ln|pow|min|max|sum|mean|norm|dot|cross|clip|clamp)\b/.test(expression)
      || /\.(?:sum|mean|norm|pow|sqrt|exp|min|max)\s*\(/.test(expression);
  }

  private renderExtractedFormulas(formulas: ExtractedFormula[]): void {
    if (formulas.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-result";
      empty.textContent = "No formulas found yet. Try code with assignments, returns, or math operations.";
      this.extractedList.replaceChildren(empty);
      return;
    }

    const fragment = document.createDocumentFragment();
    for (const formula of formulas) {
      const item = document.createElement("article");
      item.className = "formula-result";

      const code = document.createElement("code");
      code.className = "formula-text";
      code.textContent = formula.formula;

      const note = document.createElement("p");
      note.className = formula.plotExpression ? "formula-note plot-ready" : "formula-note";
      note.textContent = formula.note;

      const actions = document.createElement("div");
      actions.className = "formula-actions";

      const copy = this.createFormulaButton("Copy formula", () => {
        void this.copyText(formula.formula, copy);
      });
      actions.append(copy);

      if (formula.plotExpression) {
        const plotCode = document.createElement("code");
        plotCode.className = "plot-expression";
        plotCode.textContent = `Graph: ${formula.plotExpression}`;

        const add = this.createFormulaButton("Add to graph", () => {
          this.switchSidebarTab("expressions");
          this.addExpression(formula.plotExpression ?? "");
        });
        const copyPlot = this.createFormulaButton("Copy graph", () => {
          void this.copyText(formula.plotExpression ?? "", copyPlot);
        });

        actions.append(add, copyPlot);
        item.replaceChildren(code, plotCode, note, actions);
      } else {
        item.replaceChildren(code, note, actions);
      }
      fragment.append(item);
    }

    this.extractedList.replaceChildren(fragment);
  }

  private createFormulaButton(label: string, onClick: () => void): HTMLButtonElement {
    const button = document.createElement("button");
    button.className = "secondary-button compact";
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", onClick);
    return button;
  }

  private getRenderedFormulas(): string[] {
    return Array.from(this.extractedList.querySelectorAll<HTMLElement>(".formula-text"))
      .map((element) => element.textContent?.trim() ?? "")
      .filter(Boolean);
  }

  private async copyText(text: string, button: HTMLButtonElement): Promise<void> {
    const previousText = button.textContent ?? "Copy";
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        this.copyTextFallback(text);
      }
      button.textContent = "Copied";
    } catch {
      button.textContent = "Copy failed";
    } finally {
      window.setTimeout(() => {
        button.textContent = previousText;
      }, 1200);
    }
  }

  private copyTextFallback(text: string): void {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.append(textarea);
    textarea.focus();
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
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
