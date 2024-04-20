/** @param {NS} ns */

// 5.75 GB total RAM across 3 scripts
// 16 GB : 2 threads each (4.5 free)
// 32 GB : 5 threads each (3.25 free)
// 64 GB : 11 threads each (.75 free)
// 128 GB : 22 threads each (1.5 free)
// 256 GB : 

export async function main(ns) {
  // array of main files to copy and use
  const files = ["weaken-template.js", "hack-template.js", "grow-template.js"];
  //ns.tprint(`TEST: ns.getHostName = ${ns.getHostname()}`);

  // calculate total script ram usage
  let ram_req = 0;
  for (let i=0; i < files.length; ++i) {
    ram_req += ns.getScriptRam(files[i]);
  }
  //ns.tprint(`total script ram required: ${ram_req}\n\n`);

  // Array of all servers that don't need any ports opened
  // to gain root access.
  const servers = ["n00dles",          // 4 GB
                   "foodnstuff",       // 16 GB
                   "sigma-cosmetics",  // 16 GB
                   "joesguns",         // 16 GB
                   "nectar-net",       // 16 GB
                   "hong-fang-tea",    // 16 GB
                   "harakiri-sushi",   // 16 GB
                   "max-hardware",     // 32 GB
                   "neo-net",          // 32 GB
                   "zer0",             // 32 GB
                   "iron-gym",         // 32 GB
//                   "CSEC"];            // 8 GB
                   "phantasy",         // 32 GB
                   "omega-net",        // 32 GB
                   "silver-helix",     // 64 GB
                   "the-hub",          // 8 GB
                   "avmnite-02h",      // 64 GB
                   "johnson-ortho",    // 0? GB
                   "crush-fitness",    // 0? GB
                   "netlink",          // 64 GB
                   "computek",         // 0? GB
                   "summit-uni",       // 64 GB
                   "catalyst",         // 64 GB
                   "I.I.I.I",          // 256 GB
                   "rothman-uni",      // 32 GB
                   "syscore",          // 0? GB
                   "zb-institute"      // 64 GB
];



  // variable to see how many servers are affected
  let affect_server_count = 0;



  //ns.tprint("Beginning main loop - 0 port");
  // Copy our scripts onto each server that requires 0 ports to gain
  // root access. Then use nuke() to gain admin access and run the
  // scripts.
  for (let i = 0; i < servers.length; ++i) {
      const serv = servers[i];
      // calculate max threads
      let threads = Math.floor((ns.getServerMaxRam(serv) - ns.getServerUsedRam(serv)) / ram_req);
      if (threads <= 0) {
        threads = 1;
      }
      // bool to determine if the current server is hackable
      let pass = true;

      // announce this portion
      ns.tprint(`Next server: ${serv}. Beginning in 2s...`);
      await ns.sleep(2000);

      // check for current hack level vs. server
      if (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
        ns.tprint(`Current server ${serv} is not currently hackable\n\n`);
        pass = false;
      }
      // WHILE OPTION FOR SLEEPING INSTEAD
      //while (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
      //  await ns.sleep(60000);
      //}

      if (pass) {
        switch (serv) {
            case "n00dles":
              //await copyFiles(ns, "early-hack-template.js", serv);
              ns.scp("early-hack-template.js",serv);

              //ns.nuke(serv);
              if (!ns.hasRootAccess(serv)) {
                await access(ns,serv,ns.getServerNumPortsRequired(serv));
              }

              ns.tprint(`Launching script 'early-hack-template.js' on server '${serv}' with 1 thread`);
              ns.exec("early-hack-template.js", serv);
              ns.tprint(`early-hack-template.js successfully running on ${serv}\n\n`);
              affect_server_count++;

              break;
            
            default:
              //await copyFiles(ns, files, serv);
              if (ns.scp(files,serv)) {
                ns.tprint(`copied files to ${serv}`);
              } else {
                ns.tprint(`scp failed to copy files ${files} to server ${serv}`);
              }
              //ns.tprint(`test: files done copied to ${serv} and will run on ${threads} threads.`);

              //ns.tprint(`Nuking ${serv} in 3s...`);
              //await ns.sleep(3000);
              //ns.nuke(serv);
              if (!ns.hasRootAccess(serv)) {
                await access(ns,serv,ns.getServerNumPortsRequired(serv));
              }

              ns.tprint(`Launching scripts '${files}' on server ${serv} with ${threads} threads in 1s...`);
              await ns.sleep(1000);

              //for (let j = 0; j < files.length; ++j) {
              //  ns.exec(files[j], serv, threads);
              //  await ns.sleep(1000); // sleep for 1 second
              //}
              await execFiles(ns, files, serv, threads);
              ns.tprint(`All files successfully running on ${serv}\n\n`);
              //await ns.sleep(5000);
              affect_server_count++;
          }
      }
  }



  //ns.tprint("Beginning main loop - 1 port");
  // Copy our scripts onto each server that requires 1 port and 32 GB
  // to gain root access. Then use brutessh() and nuke() to gain
  // admin access and run the scripts.
  for (let i = 0; i < servers1Port.length; ++i) {
      const serv = servers1Port[i];
      // calculate max threads
      let threads = Math.floor((ns.getServerMaxRam(serv) - ns.getServerUsedRam(serv)) / ram_req);
      if (threads <= 0) {
        threads = 1;
      }
      // bool to determine if the current server is hackable
      let pass = true;

      // announce this portion
      ns.tprint(`Next server: ${serv}. Beginning in 2s...`);
      await ns.sleep(2000);

      // check for current hack level vs. server
      if (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
        ns.tprint(`Current server ${serv} is not currently hackable\n\n`);
        pass = false;
      }
      // WHILE OPTION FOR SLEEPING INSTEAD
      //while (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
      //  await ns.sleep(60000);
      //}

      if (pass) {
        if (ns.scp(files,serv)) {
          ns.tprint(`copied files to ${serv}`);
        } else {
          ns.tprint(`scp failed to copy files ${files} to server ${serv}`);
        }
        //ns.tprint(`test: files done copied to ${serv} and will run on ${threads} threads.`);

        //ns.tprint(`BruteSSH-ing ${serv} in 3s...`);
        //await ns.sleep(3000);
        //ns.brutessh(serv);
        //ns.tprint(`Nuking ${serv} in 3s...`);
        //await ns.sleep(3000);
        //ns.nuke(serv);
        if (!ns.hasRootAccess(serv)) {
          await access(ns,serv,ns.getServerNumPortsRequired(serv));
        }

        ns.tprint(`Launching scripts '${files}' on server ${serv} with ${threads} threads in 1s...`);
        await ns.sleep(1000);

        await execFiles(ns, files, serv, threads);
        ns.tprint(`All files successfully running on ${serv}\n\n`);
        //await ns.sleep(5000);
        affect_server_count++;
      }
  }


  
  // Wait until we acquire the "FTPCrack.exe" program
  //while (!ns.fileExists("FTPCrack.exe", "home")) {
  //    await ns.sleep(60000);
  //}
  // IF VARIANT
  if (!ns.fileExists("FTPCrack.exe", "home")) {
      ns.tprint("File FTPCrack.exe does not exist.");
      ns.tprint(`Affected servers: ${affect_server_count}`);
      return;
  }



  //ns.tprint("Beginning main loop - 2 port");
  // Copy our scripts onto each server that requires 2 ports to gain
  // root access. Then use brutessh() and nuke() and ftpcrack() to
  // gain admin access and run the scripts.
  for (let i = 0; i < servers2Port.length; ++i) {
      const serv = servers2Port[i];
      // calculate max threads
      let threads = Math.floor((ns.getServerMaxRam(serv) - ns.getServerUsedRam(serv)) / ram_req);
      if (threads <= 0) {
        threads = 1;
      }
      // bool to determine if the current server is hackable
      let pass = true;

      // announce this portion
      ns.tprint(`Next server: ${serv}. Beginning in 2s...`);
      await ns.sleep(2000);

      // check for current hack level vs. server
      if (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
        ns.tprint(`Current server ${serv} is not currently hackable\n\n`);
        pass = false;
      }
      // WHILE OPTION FOR SLEEPING INSTEAD
      //while (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
      //  await ns.sleep(60000);
      //}

      if (pass) {
        if (ns.scp(files,serv)) {
          ns.tprint(`copied files to ${serv}`);
        } else {
          ns.tprint(`scp failed to copy files ${files} to server ${serv}`);
        }
        //ns.tprint(`test: files done copied to ${serv} and will run on ${threads} threads.`);

        if (!ns.hasRootAccess(serv)) {
          await access(ns,serv,ns.getServerNumPortsRequired(serv));
        }

        ns.tprint(`Launching scripts '${files}' on server ${serv} with ${threads} threads in 1s`);
        await ns.sleep(1000);

        await execFiles(ns, files, serv, threads);
        ns.tprint(`All files successfully running on ${serv}\n\n`);
        //await ns.sleep(5000);
        affect_server_count++;
      }
  }



  // Wait until we acquire the "RelaySMTP.exe" program
  //while (!ns.fileExists("RelaySMTP.exe", "home")) {
  //    await ns.sleep(60000);
  //}
  // IF VARIANT
  if (!ns.fileExists("RelaySMTP.exe", "home")) {
      ns.tprint("File RelaySMTP.exe does not exist.");
      ns.tprint(`Affected servers: ${affect_server_count}`);
      return;
  }



  //ns.tprint("Beginning main loop - 3 port");
  // Copy our scripts onto each server that requires 3 ports to gain
  // root access. Then use ftpcrack(), brutessh(), relaysmtp, and
  // nuke() to gain admin access and run the scripts.
  for (let i = 0; i < servers3Port.length; ++i) {
      const serv = servers3Port[i];
      // calculate max threads
      let threads = Math.floor((ns.getServerMaxRam(serv) - ns.getServerUsedRam(serv)) / ram_req);
      if (threads <= 0) {
        threads = 1;
      }
      // bool to determine if the current server is hackable
      let pass = true;

      // announce this portion
      ns.tprint(`Next server: ${serv}. Beginning in 2s...`);
      await ns.sleep(2000);

      // check for current hack level vs. server
      if (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
        ns.tprint(`Current server ${serv} is not currently hackable\n\n`);
        pass = false;
      }
      // WHILE OPTION FOR SLEEPING INSTEAD
      //while (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
      //  await ns.sleep(60000);
      //}

      if (pass) {
        if (ns.scp(files,serv)) {
          ns.tprint(`copied files to ${serv}`);
        } else {
          ns.tprint(`scp failed to copy files ${files} to server ${serv}`);
        }
        //ns.tprint(`test: files done copied to ${serv} and will run on ${threads} threads.`);

        if (!ns.hasRootAccess(serv)) {
          await access(ns,serv,ns.getServerNumPortsRequired(serv));
        }

        ns.tprint(`Launching scripts '${files}' on server ${serv} with ${threads} threads in 1s`);
        await ns.sleep(1000);

        await execFiles(ns, files, serv, threads);
        ns.tprint(`All files successfully running on ${serv}\n\n`);
        //await ns.sleep(5000);
        affect_server_count++;
      }
  }



  // Wait until we acquire the "HTTPWorm.exe" program
  //while (!ns.fileExists("HTTPWorm.exe", "home")) {
  //    await ns.sleep(60000);
  //}
  // IF VARIANT
  if (!ns.fileExists("HTTPWorm.exe", "home")) {
      ns.tprint("File HTTPWorm.exe does not exist.");
      ns.tprint(`Affected servers: ${affect_server_count}`);
      return;
  }



  //ns.tprint("Beginning main loop - 4 port");
  // Copy our scripts onto each server that requires 3 ports to gain
  // root access. Then use ftpcrack(), brutessh(), relaysmtp(),
  // httpworm(), and nuke() to gain admin access and run the scripts.
  for (let i = 0; i < servers4Port.length; ++i) {
      const serv = servers4Port[i];
      // calculate max threads
      let threads = Math.floor((ns.getServerMaxRam(serv) - ns.getServerUsedRam(serv)) / ram_req);
      if (threads <= 0) {
        threads = 1;
      }
      // bool to determine if the current server is hackable
      let pass = true;

      // announce this portion
      ns.tprint(`Next server: ${serv}. Beginning in 2s...`);
      await ns.sleep(2000);

      // check for current hack level vs. server
      if (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
        ns.tprint(`Current server ${serv} is not currently hackable\n\n`);
        pass = false;
      }
      // WHILE OPTION FOR SLEEPING INSTEAD
      //while (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
      //  await ns.sleep(60000);
      //}

      if (pass) {
        if (ns.scp(files,serv)) {
          ns.tprint(`copied files to ${serv}`);
        } else {
          ns.tprint(`scp failed to copy files ${files} to server ${serv}`);
        }
        //ns.tprint(`test: files done copied to ${serv} and will run on ${threads} threads.`);

        if (!ns.hasRootAccess(serv)) {
          await access(ns,serv,ns.getServerNumPortsRequired(serv));
        }

        ns.tprint(`Launching scripts '${files}' on server ${serv} with ${threads} threads in 1s`);
        await ns.sleep(1000);

        await execFiles(ns, files, serv, threads);
        ns.tprint(`All files successfully running on ${serv}\n\n`);
        //await ns.sleep(5000);
        affect_server_count++;
      }
  }



  // Wait until we acquire the "SQLInject.exe" program
  //while (!ns.fileExists("SQLInject.exe", "home")) {
  //    await ns.sleep(60000);
  //}
  // IF VARIANT
  if (!ns.fileExists("SQLInject.exe", "home")) {
      ns.tprint("File SQLInject.exe does not exist.");
      ns.tprint(`Affected servers: ${affect_server_count}`);
      return;
  }



  //ns.tprint("Beginning main loop - 5 port");
  // Copy our scripts onto each server that requires 3 ports to gain
  // root access. Then use ftpcrack(), brutessh(), relaysmtp, and
  // nuke() to gain admin access and run the scripts.
  for (let i = 0; i < servers5Port.length; ++i) {
      const serv = servers5Port[i];
      // calculate max threads
      let threads = Math.floor((ns.getServerMaxRam(serv) - ns.getServerUsedRam(serv)) / ram_req);
      if (threads <= 0) {
        threads = 1;
      }
      // bool to determine if the current server is hackable
      let pass = true;

      // announce this portion
      ns.tprint(`Next server: ${serv}. Beginning in 2s...`);
      await ns.sleep(2000);

      // check for current hack level vs. server
      if (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
        ns.tprint(`Current server ${serv} is not currently hackable\n\n`);
        pass = false;
      }
      // WHILE OPTION FOR SLEEPING INSTEAD
      //while (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
      //  await ns.sleep(60000);
      //}

      if (pass) {
        if (ns.scp(files,serv)) {
          ns.tprint(`copied files to ${serv}`);
        } else {
          ns.tprint(`scp failed to copy files ${files} to server ${serv}`);
        }
        //ns.tprint(`test: files done copied to ${serv} and will run on ${threads} threads.`);

        if (!ns.hasRootAccess(serv)) {
          await access(ns,serv,ns.getServerNumPortsRequired(serv));
        }

        ns.tprint(`Launching scripts '${files}' on server ${serv} with ${threads} threads in 1s`);
        await ns.sleep(1000);

        await execFiles(ns, files, serv, threads);
        ns.tprint(`All files successfully running on ${serv}\n\n`);
        //await ns.sleep(5000);
        affect_server_count++;
      }
  }

  ns.tprint("DONE -- copying/executing scripts");
  ns.tprint(`Affected servers: ${affect_server_count}`);
  return affect_server_count;
}



// from ChatGPT
async function execFiles(ns, files, target, threads) {
  return new Promise(resolve => {
      const executeFile = async (fileIndex) => {
          if (fileIndex >= files.length) {
              resolve(true); // All files executed successfully
              return;
          }
          
          const file = files[fileIndex];
          // successful start
          if (ns.exec(file, target, threads)) {
              ns.tprint(`File ${file} running on ${target}`);
              setTimeout(() => executeFile(fileIndex + 1), 1000); // Execute next file after 1 second
          }
          // could not execute file
          else {
              ns.tprint(`Failed to start file ${file} on server ${target}`);
              setTimeout(() => executeFile(fileIndex), 1000); // Retry current file after 1 second
          }
      };

      executeFile(0); // Start executing files from the beginning of the array
  });
}



async function access(ns, server, num_ports) {
  return new Promise(async resolve => {
      switch (num_ports) {
          case 0:
              ns.tprint(`Nuking ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.nuke(server);

              resolve(true);
              break;
          case 1:
              // Wait until we acquire the "BruteSSH.exe" program
              //while (!ns.fileExists("BruteSSH.exe", "home")) {
              //    await ns.sleep(60000);
              //}
              // IF VARIANT
              if (!ns.fileExists("BruteSSH.exe", "home")) {
                ns.tprint("File BruteSSH.exe does not exist.");
                //ns.tprint(`Affected servers: ${affect_server_count}`);
                return;
              }

              ns.tprint(`BruteSSH-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.brutessh(server);
              
              ns.tprint(`Nuking ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.nuke(server);

              resolve(true);
              break;
          case 2:
              ns.tprint(`FTPCrack-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.ftpcrack(server);

              ns.tprint(`BruteSSH-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.brutessh(server);

              ns.tprint(`Nuking ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.nuke(server);

              resolve(true);
              break;
          case 3:
              ns.tprint(`RelaySMTP-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.relaysmtp(server);
              
              ns.tprint(`FTPCrack-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.ftpcrack(server);

              ns.tprint(`BruteSSH-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.brutessh(server);

              ns.tprint(`Nuking ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.nuke(server);

              resolve(true);
              break;
          case 4:
              ns.tprint(`HTTPWorm-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.httpworm(server);

              ns.tprint(`RelaySMTP-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.relaysmtp(server);
              
              ns.tprint(`FTPCrack-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.ftpcrack(server);

              ns.tprint(`BruteSSH-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.brutessh(server);

              ns.tprint(`Nuking ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.nuke(server);

              resolve(true);
              break;
          case 5:
              ns.tprint(`SQLInject-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.sqlinject(server);
              
              ns.tprint(`HTTPWorm-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.httpworm(server);

              ns.tprint(`RelaySMTP-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.relaysmtp(server);
              
              ns.tprint(`FTPCrack-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.ftpcrack(server);

              ns.tprint(`BruteSSH-ing ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.brutessh(server);

              ns.tprint(`Nuking ${server} (${num_ports} ports) in 3s...`);
              await ns.sleep(3000);
              ns.nuke(server);

              resolve(true);
              break;
      }
  });
}