/** @param {NS} ns */
export async function main(ns, servers) {
  for (let i = 0; i < servers.length; ++i) {
    ns.killall(servers[i]);
  }
  ns.tprint("DONE -- stopping scripts");
}

