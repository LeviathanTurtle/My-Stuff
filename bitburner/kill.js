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
      ns.killall(servers[i]);
    }
    ns.tprint("DONE -- stopping scripts");
}

