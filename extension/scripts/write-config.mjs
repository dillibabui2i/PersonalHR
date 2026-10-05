import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const envFile = readFileSync(new URL("../../server/.env", import.meta.url), "utf8");
const match = envFile.match(/^EXTENSION_KEY=(.*)$/m);
const extensionKey = (match?.[1] ?? "").trim().replace(/^['"]|['"]$/g, "");
if (extensionKey === "") {
  console.error("EXTENSION_KEY is missing from server/.env");
  process.exit(1);
}

const distUrl = new URL("../dist/", import.meta.url);
mkdirSync(distUrl, { recursive: true });
writeFileSync(new URL("./extension-key.txt", distUrl), extensionKey);
