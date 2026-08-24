// Build the write-up as a .docx for import into Google Docs.
//
//   node scripts/make_docx.js
//
// Google Docs is the target, so: US Letter, table widths in DXA on both the table and
// every cell (PERCENTAGE breaks on import), ShadingType.CLEAR, and built-in heading
// levels so the outline pane works.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, ExternalHyperlink,
  LevelFormat, convertInchesToTwip,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const SOURCE = path.join(ROOT, "WRITEUP_DRAFT.md");
const OUT = path.join(ROOT, "results", "Model_Forensics_SPAR_take-home_Mohit_Goyal.docx");

const CONTENT_DXA = 9360;           // Letter, 12240 minus 1 inch margins each side
const INK = "1A1A1A";
const SOFT = "5A5A5A";
const RULE = "D4D4D4";
const HEADER_FILL = "F1EFEC";

// --- inline formatting ------------------------------------------------------ //

// Splits on **bold**, `code`, and [text](url), leaving plain text between.
const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;

function runs(text, base = {}) {
  const out = [];
  for (const piece of text.split(INLINE)) {
    if (!piece) continue;
    if (piece.startsWith("**") && piece.endsWith("**")) {
      out.push(new TextRun({ ...base, text: piece.slice(2, -2), bold: true }));
    } else if (piece.startsWith("`") && piece.endsWith("`")) {
      out.push(new TextRun({ ...base, text: piece.slice(1, -1), font: "Menlo", size: 19 }));
    } else if (piece.startsWith("[")) {
      const m = piece.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      out.push(new ExternalHyperlink({
        link: m[2],
        children: [new TextRun({ ...base, text: m[1], style: "Hyperlink" })],
      }));
    } else {
      out.push(new TextRun({ ...base, text: piece }));
    }
  }
  return out.length ? out : [new TextRun({ ...base, text: "" })];
}

function stripInline(text) {
  return text
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
}

// --- block builders --------------------------------------------------------- //

function heading(text, level) {
  return new Paragraph({
    children: runs(text, { color: INK }),
    heading: level,
    spacing: { before: level === HeadingLevel.HEADING_1 ? 360 : 260, after: 130 },
    keepNext: true,
  });
}

function body(text) {
  return new Paragraph({
    children: runs(text, { color: INK, size: 21 }),
    spacing: { after: 150, line: 300 },
  });
}

function bullet(text, ordered) {
  return new Paragraph({
    children: runs(text, { color: INK, size: 21 }),
    spacing: { after: 90, line: 290 },
    ...(ordered
      ? { numbering: { reference: "ordered", level: 0 } }
      : { bullet: { level: 0 } }),
  });
}

function rule() {
  return new Paragraph({
    text: "",
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 8 } },
    spacing: { before: 200, after: 240 },
  });
}

function cell(text, { header = false, width }) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header
      ? { type: ShadingType.CLEAR, fill: HEADER_FILL, color: "auto" }
      : undefined,
    margins: { top: 90, bottom: 90, left: 130, right: 130 },
    children: [new Paragraph({
      children: runs(text, { color: header ? INK : SOFT, size: 19, bold: header || undefined }),
      spacing: { after: 0, line: 260 },
    })],
  });
}

function table(rows) {
  const columns = rows[0].length;
  // First column carries the labels, so give it the slack; the rest share evenly.
  const firstShare = columns <= 3 ? 0.44 : 0.30;
  const first = Math.round(CONTENT_DXA * firstShare);
  const rest = Math.floor((CONTENT_DXA - first) / (columns - 1));
  const widths = [CONTENT_DXA - rest * (columns - 1), ...Array(columns - 1).fill(rest)];

  return new Table({
    width: { size: CONTENT_DXA, type: WidthType.DXA },
    columnWidths: widths,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: RULE },
      insideVertical: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    },
    rows: rows.map((cells, r) => new TableRow({
      tableHeader: r === 0,
      children: cells.map((text, c) => cell(text, { header: r === 0, width: widths[c] })),
    })),
  });
}

function figure(file, alt) {
  const data = fs.readFileSync(path.join(ROOT, file));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 90 },
      children: [new ImageRun({
        type: "png",
        data,
        transformation: { width: 624, height: 221 },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [new TextRun({ text: alt, italics: true, color: SOFT, size: 18 })],
    }),
  ];
}

// --- markdown walk ---------------------------------------------------------- //

function parse(markdown) {
  const lines = markdown.split("\n");
  const blocks = [];
  let i = 0;

  const isTableRow = (l) => l.trim().startsWith("|") && l.trim().endsWith("|");
  const splitRow = (l) => l.trim().slice(1, -1).split("|").map((c) => c.trim());

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) { i++; continue; }

    if (/^!\[/.test(trimmed)) {
      const m = trimmed.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
      blocks.push(...figure(m[2], m[1]));
      i++; continue;
    }
    if (/^---+$/.test(trimmed)) { blocks.push(rule()); i++; continue; }
    if (trimmed.startsWith("### ")) { blocks.push(heading(trimmed.slice(4), HeadingLevel.HEADING_2)); i++; continue; }
    if (trimmed.startsWith("## ")) { blocks.push(heading(trimmed.slice(3), HeadingLevel.HEADING_1)); i++; continue; }
    if (trimmed.startsWith("# ")) {
      blocks.push(new Paragraph({
        children: runs(trimmed.slice(2), { color: INK }),
        heading: HeadingLevel.TITLE,
        spacing: { after: 120 },
      }));
      i++; continue;
    }
    if (trimmed.startsWith("```")) {
      i++;
      const code = [];
      while (i < lines.length && !lines[i].trim().startsWith("```")) code.push(lines[i++]);
      i++;
      for (const c of code) {
        blocks.push(new Paragraph({
          shading: { type: ShadingType.CLEAR, fill: "F5F4F2", color: "auto" },
          spacing: { after: 0, line: 260 },
          children: [new TextRun({ text: c || " ", font: "Menlo", size: 18, color: SOFT })],
        }));
      }
      blocks.push(new Paragraph({ text: "", spacing: { after: 150 } }));
      continue;
    }
    if (isTableRow(line)) {
      const rows = [];
      while (i < lines.length && isTableRow(lines[i])) {
        const cells = splitRow(lines[i]);
        if (!cells.every((c) => /^:?-+:?$/.test(c) || c === "")) rows.push(cells);
        i++;
      }
      blocks.push(table(rows));
      blocks.push(new Paragraph({ text: "", spacing: { after: 200 } }));
      continue;
    }
    if (/^[-*] /.test(trimmed)) { blocks.push(bullet(trimmed.slice(2), false)); i++; continue; }
    if (/^\d+\. /.test(trimmed)) { blocks.push(bullet(trimmed.replace(/^\d+\.\s+/, ""), true)); i++; continue; }

    // a paragraph runs until a blank line
    const para = [trimmed];
    i++;
    while (i < lines.length && lines[i].trim() && !isTableRow(lines[i])
           && !/^([-*] |\d+\. |#|```|---+$|!\[)/.test(lines[i].trim())) {
      para.push(lines[i].trim());
      i++;
    }
    blocks.push(body(para.join(" ")));
  }
  return blocks;
}

// --- document --------------------------------------------------------------- //

const children = parse(fs.readFileSync(SOURCE, "utf8"));

const doc = new Document({
  creator: "Mohit Goyal",
  title: "Model Forensics SPAR take-home",
  description: "Why Claude 4.5 refuses benign safety research: two mechanisms, not one",
  numbering: {
    config: [{
      reference: "ordered",
      levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.START,
        style: { paragraph: { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.22) } } },
      }],
    }],
  },
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 21, color: INK } },
      title: { run: { font: "Calibri", size: 40, bold: true, color: INK } },
      heading1: { run: { font: "Calibri", size: 28, bold: true, color: INK } },
      heading2: { run: { font: "Calibri", size: 23, bold: true, color: INK } },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, buffer);
  console.log(`wrote ${OUT} (${(buffer.length / 1024).toFixed(0)} KB, ${children.length} blocks)`);
});
