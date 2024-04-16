/** @param {NS} ns */
export async function main(ns, target) {
  // Defines the "target server", which is the server
  // that we're going to hack. In this case, it's "n00dles"
  //const target = ns.getHostname();
  //const target = ns.args.length > 0 ? ns.args[0] : ns.getHostname();
  target = ns.args.length > 0 ? ns.args[0] : ns.getHostname();

  // Defines how much money a server should have before we hack it
  // In this case, it is set to the maximum amount of money.
  const moneyThresh = ns.getServerMaxMoney(target);

  // If we have the BruteSSH.exe program, use it to open the SSH Port
  // on the target server
  //if (ns.fileExists("BruteSSH.exe", "home")) {
  //    ns.brutessh(target);
  //}

  // Get root access to target server
  //ns.nuke(target);

  // Infinite loop that continously hacks/grows/weakens the target server
  while(true) {
    if (ns.getServerMoneyAvailable(target) < moneyThresh) {
        // If the server's money is less than our threshold, grow it
        await ns.grow(target);
        //await ns.sleep(3000);
    } 
  }
}