/** @param {NS} ns */
export async function main(ns) {
    // ram req: 
    ns.exec("kill.js","home");
    await ns.sleep(1000);
  
  
    // ram req: 
    ns.exec("clean.js","home");
    await ns.sleep(1000);
  
  
    // ram req: 
    ns.exec("script_startup.js","home");
    await ns.sleep(1000);


    // ram req: 
    ns.exec("home-script_startup.js","home");
    await ns.sleep(1000);
  }