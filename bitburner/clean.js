/** @param {NS} ns */
export async function main(ns) {
    // Array of all servers that have the main .js files
    const servers = ["n00dles",
                     "foodnstuff",
                     "sigma-cosmetics",
                     "joesguns",
//                     "nectar-net",
                     "hong-fang-tea",
                     "harakiri-sushi",
//                     "max-hardware",
//                     "neo-net",
//                     "zer0",
                     "iron-gym",
//                     "phantasy",
//                     "omega-net",
//                     "silver-helix",
//                     "avmnite-02h"
                     ];

    for (let i = 0; i < servers.length; ++i) {
      ns.tprint(`Deleting weaken-template.js from ${servers[i]}`);
      ns.rm("weaken-template.js",servers[i]);
      //await ns.sleep(250);

      ns.tprint(`Deleting grow-template.js from ${servers[i]}`);
      ns.rm("grow-template.js",servers[i]);
      //await ns.sleep(250);

      ns.tprint(`Deleting hack-template.js from ${servers[i]}`);
      ns.rm("hack-template.js",servers[i]);
      //await ns.sleep(250);
    }

}