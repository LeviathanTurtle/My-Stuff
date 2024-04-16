/** @param {NS} ns */
export async function main(ns, target) {
  // Defines the "target server", which is the server
  // that we're going to hack. In this case, it's "n00dles"
  //const target = ns.getHostname();
  //const target = ns.args.length > 0 ? ns.args[0] : ns.getHostname();
  target = ns.args.length > 0 ? ns.args[0] : ns.getHostname();

  // If we have the BruteSSH.exe program, use it to open the SSH Port
  // on the target server
  //if (ns.fileExists("BruteSSH.exe", "home")) {
  //    ns.brutessh(target);
  //}

  // Get root access to target server
  //ns.nuke(target);

  // Infinite loop that continously hacks/grows/weakens the target server
  while(true) {
    await ns.hack(target);
    //await ns.sleep(3000);
  }
}