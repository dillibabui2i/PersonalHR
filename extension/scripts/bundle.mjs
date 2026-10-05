import { copyFileSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";

const dist = new URL("../dist/", import.meta.url);
const extensionKey = readFileSync(new URL("./extension-key.txt", dist), "utf8");
const background = readFileSync(new URL("./background.js", dist), "utf8").replaceAll(
  "__EXTENSION_KEY__",
  extensionKey,
);
writeFileSync(new URL("./background.js", dist), background);
copyFileSync(new URL("../manifest.json", import.meta.url), new URL("./manifest.json", dist));
copyFileSync(new URL("../styles.css", import.meta.url), new URL("./styles.css", dist));
mkdirSync(new URL("./icons/", dist), { recursive: true });
for (const icon of ["icon16.png", "icon32.png", "icon48.png", "icon128.png", "ideas2it-mark.png"]) {
  copyFileSync(new URL(`../icons/${icon}`, import.meta.url), new URL(`./icons/${icon}`, dist));
}
unlinkSync(new URL("./extension-key.txt", dist));
for (const leftover of ["generated-config.js", "config.js"]) {
  try {
    unlinkSync(new URL(`./${leftover}`, dist));
  } catch {
    continue;
  }
}
