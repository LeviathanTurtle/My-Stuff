/** @param {NS} ns */
export async function main(ns, servers) {
  for (let i = 0; i < servers.length; ++i) {
    // these are in if statements to avoid deleting files
    // that aren't actually there
    if (ns.fileExists("weaken-template.js",servers[i])) {
      ns.tprint(`Deleting weaken-template.js from ${servers[i]}`);
      ns.rm("weaken-template.js",servers[i]);
      //await ns.sleep(250);
    }

    if (ns.fileExists("grow-template.js",servers[i])) {
      ns.tprint(`Deleting grow-template.js from ${servers[i]}`);
      ns.rm("grow-template.js.js",servers[i]);
      //await ns.sleep(250);
    }

    if (ns.fileExists("hack-template.js",servers[i])) {
      ns.tprint(`Deleting hack-template.js from ${servers[i]}`);
      ns.rm("hack-template.js.js",servers[i]);
      //await ns.sleep(250);
    }
  }
  ns.tprint("DONE -- deleting scripts");
}