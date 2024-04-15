/** @param {NS} ns */
export async function main(ns) {
  // Array of all servers that have the main .js files
  const servers = ["n00dles",
                   "foodnstuff",
                   "sigma-cosmetics",
                   "joesguns",
                   "nectar-net",
                   "hong-fang-tea",
                   "harakiri-sushi",
                   "max-hardware",
                   "neo-net",
                   "zer0",
                   "iron-gym",
                   "CSEC",
                   "phantasy",
                   "omega-net",
                   "silver-helix",
                   "the-hub",
                   "avmnite-02h",
                   "johnson-ortho",
                   "crush-fitness",
                   "netlink",
                   "computek",
                   "summit-uni",
                   "catalyst",
                   "I.I.I.I",
                   "rothman-uni",
                   "syscore",
                   "zb-institute"
                   ];

  for (let i = 0; i < servers.length; ++i) {
    // these are in if statements to avoid deleting files
    // that aren't actually there
    if (ns.fileExists("weaken-template.js",servers[i])) {
      ns.tprint(`Deleting weaken-template.js from ${servers[i]}`);
      ns.rm("weaken-template.js",servers[i]);
      //await ns.sleep(250);
    }

    if (ns.fileExists("grow-template.js.js",servers[i])) {
      ns.tprint(`Deleting grow-template.js.js from ${servers[i]}`);
      ns.rm("grow-template.js.js",servers[i]);
      //await ns.sleep(250);
    }

    if (ns.fileExists("hack-template.js.js",servers[i])) {
      ns.tprint(`Deleting hack-template.js.js from ${servers[i]}`);
      ns.rm("hack-template.js.js",servers[i]);
      //await ns.sleep(250);
    }
  }
  ns.tprint("DONE -- deleting scripts");
}