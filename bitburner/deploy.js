/** @param {NS} ns */
export async function main(ns) {
    // optional help flag
      const args = ns.flags([["help", false]]);
    // if help flag specified or incorrect usage
      if (args.help || args._.length < 2) {
          ns.tprint("This script deploys another script on a server with maximum threads possible.");
          ns.tprint(`Usage: run ${ns.getScriptName()} HOST SCRIPT ARGUMENTS`);
          ns.tprint("Example:");
          ns.tprint(`> run ${ns.getScriptName()} n00dles basic_hack.js foodnstuff`);
          return;
      }
  
    // host is first arg
      const host = args._[0];
    // script in question is second arg
      const script = args._[1];
    // any other args
      const script_args = args._.slice(2);
  
    // check if the server exists
      if (!ns.serverExists(host)) {
          ns.tprint(`Server '${host}' does not exist. Aborting.`);
          return;
      }
    // check if the script exists
      if (!ns.ls(ns.getHostname()).find(f => f === script)) {
          ns.tprint(`Script '${script}' does not exist. Aborting.`);
          return;
      }
  
    // calculate max threads
      const threads = Math.floor((ns.getServerMaxRam(host) - ns.getServerUsedRam(host)) / ns.getScriptRam(script));
      ns.tprint(`Launching script '${script}' on server '${host}' with ${threads} threads and the following arguments: ${script_args}`);
      await ns.scp(script, ns.getHostname(), host);
      ns.exec(script, host, threads, ...script_args);
  }