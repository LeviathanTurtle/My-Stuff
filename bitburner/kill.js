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
//                     "CSEC",
                   "phantasy",
                   "omega-net",
                   "silver-helix",
                   "the-hub",
//                     "avmnite-02h",
                   "johnson-ortho",
                   "crush-fitness",
                   "netlink",
                   "computek",
                   "summit-uni",
                   "catalyst",
//                     "I.I.I.I",
                   "rothman-uni",
                   "syscore",
                   "zb-institute"
                   ];

  for (let i = 0; i < servers.length; ++i) {
    switch (servers[i]) {
      case "home":
        //const master = ns.getRunningScript("master.js","home");
        //if (!master) {
        //  ns.killall(servers[i]);
        //}
        //ns.scriptKill("script_startup.js","home");

        //for (let j = 0; j < servers.length; ++j) {
        //  ns.scriptKill(`weaken-template.js ["${servers[j]}"]`);
        //  ns.scriptKill("hack-template.js");
        //  ns.scriptKill("grow-template.js");
        //}

        // from ChatGPT
        const scripts = ns.ps("home");

        // Find the script started by one of the main scripts
        const startupScript_1 = scripts.find(script => script.filename === "master.js");
        const startupScript_2 = scripts.find(script => script.filename === "kill.js");

        if (startupScript_1 || startupScript_2) {
          // Kill all other scripts except the one started by "run master.js"
          for (const script of scripts) {
            if (script.pid !== startupScript_1.pid || script.pid !== startupScript_2.pid) {
              await ns.kill(script.pid);
            }
          }
          ns.tprint("All scripts except master.js killed.");
        } else {
          ns.tprint("master.js is not running on the server.");
        }

        break;

      default:
        ns.killall(servers[i]);
    }
  }

  ns.tprint("DONE -- stopping scripts");
}

