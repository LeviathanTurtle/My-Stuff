/** @param {NS} ns */
export async function main(ns) {
    ns.exec("kill.js","home");
    await ns.sleep(1000);
  
  
    ns.exec("clean.js","home");
    await ns.sleep(1000);
  
  
    ns.exec("script_startup.js","home");
    await ns.sleep(1000);
  }