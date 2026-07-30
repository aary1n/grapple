import React from "react";
import {
  Document,
  Page,
  View,
  Text,
  Image,
  StyleSheet,
  renderToFile,
} from "@react-pdf/renderer";
import path from "path";
import fs from "fs";

const ELEMENTS = path.join(__dirname, "..", "elements");

// Windows absolute paths (C:\...) are parsed as URLs by react-pdf's image
// resolver — embed the PNGs as data URIs instead.
function element(name: string): string {
  const buf = fs.readFileSync(path.join(ELEMENTS, name));
  return `data:image/png;base64,${buf.toString("base64")}`;
}

// Nature double column: 519pt (183mm)
const WIDTH = 519;

const styles = StyleSheet.create({
  page: {
    backgroundColor: "#FFFFFF",
    fontFamily: "Helvetica",
  },
  row: {
    flexDirection: "row",
    alignItems: "flex-end",
  },
  panel: {
    position: "relative",
  },
  label: {
    position: "absolute",
    top: 0,
    left: 0,
    fontSize: 12,
    fontFamily: "Helvetica-Bold",
  },
  img: {
    objectFit: "contain",
  },
});

const LABEL_H = 14; // space reserved above each image for the panel label

function Panel({
  label,
  src,
  width,
  height,
}: {
  label: string;
  src: string;
  width: number;
  height: number;
}) {
  return (
    <View style={[styles.panel, { width, height: height + LABEL_H }]}>
      <Text style={styles.label}>{label}</Text>
      <View
        style={{
          marginTop: LABEL_H,
          width: "100%",
          height,
          justifyContent: "flex-end",
        }}
      >
        <Image style={[styles.img, { maxHeight: height }]} src={src} />
      </View>
    </View>
  );
}

// ── Figure 1: quantization (3 panels in a row) ──────────────────────────────
const F1_PANEL_H = 150;
const F1_H = F1_PANEL_H + LABEL_H + 16;

const Figure1 = (
  <Document>
    <Page size={[WIDTH, F1_H]} style={styles.page}>
      <View style={[styles.row, { padding: 8, justifyContent: "space-between" }]}>
        <Panel
          label="A"
          src={element("q_sensitivity.png")}
          width={172}
          height={F1_PANEL_H}
        />
        <Panel
          label="B"
          src={element("q_latency.png")}
          width={172}
          height={F1_PANEL_H}
        />
        <Panel
          label="C"
          src={element("q_parity.png")}
          width={150}
          height={F1_PANEL_H}
        />
      </View>
    </Page>
  </Document>
);

// ── Figure 2: semantic convergence (4 panels in a row) ──────────────────────
const F2_PANEL_H = 120;
const F2_H = F2_PANEL_H + LABEL_H + 16;

const Figure2 = (
  <Document>
    <Page size={[WIDTH, F2_H]} style={styles.page}>
      <View style={[styles.row, { padding: 8, justifyContent: "space-between" }]}>
        {["s_nll.png", "s_mse.png", "s_acc.png", "s_entropy.png"].map(
          (name, i) => (
            <Panel
              key={name}
              label={String.fromCharCode(65 + i)}
              src={element(name)}
              width={122}
              height={F2_PANEL_H}
            />
          ),
        )}
      </View>
    </Page>
  </Document>
);

const OUT = path.join(__dirname, "..");
await renderToFile(Figure1, path.join(OUT, "fig1_quantization.pdf"));
await renderToFile(Figure2, path.join(OUT, "fig2_semantic.pdf"));
console.log("rendered fig1_quantization.pdf, fig2_semantic.pdf");
