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
    // to gain root access. These have 4-16 GB of RAM
    const servers0Port = ["n00dles",
                          "foodnstuff",
                          "sigma-cosmetics",
                          "joesguns",
                          "nectar-net",
                          "hong-fang-tea",
                          "harakiri-sushi"];

    // Array of all servers that only need 1 port opened
    // to gain root access. These have 32 GB of RAM
    const servers1Port = ["max-hardware",
                          "neo-net",
                          "zer0",
                          "iron-gym"];
    
    // Array of all servers that only need 2 ports opened
    // to gain root access. These have 32-128 GB of RAM
    const servers2Port = ["phantasy",
                          "omega-net",
                          "silver-helix",
                          "avmnite-02h"];



    //ns.tprint("Beginning main loop - 0 port 16 gb");
    // Copy our scripts onto each server that requires 0 ports to gain
    // root access. Then use nuke() to gain admin access and run the
    // scripts.
    for (let i = 0; i < servers0Port.length; ++i) {
        const serv = servers0Port[i];
        // calculate max threads
        const threads = Math.floor((ns.getServerMaxRam(serv) - ns.getServerUsedRam(serv)) / ram_req);
        // bool to determine if the current server is hackable
        let pass = true;

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
                // announce this portion
                ns.tprint(`Next server: ${serv}. Beginning in 3s...`);
                await ns.sleep(3000);

                //await copyFiles(ns, "early-hack-template.js", serv);
                ns.scp("early-hack-template.js",serv);

                //ns.nuke(serv);
                await access(ns, ns.getServerNumPortsRequired(serv));

                ns.tprint(`Launching script 'early-hack-template.js' on server '${serv}' with 1 thread`);
                ns.exec("early-hack-template.js", serv);
                ns.tprint(`early-hack-template.js successfully running on ${serv}\n\n`);

                break;
              
              default:
                // announce this portion
                ns.tprint(`Next server: ${serv}. Beginning in 5s...`);
                await ns.sleep(5000);

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
                await access(ns, ns.getServerNumPortsRequired(serv));

                ns.tprint(`Launching scripts '${files}' on server ${serv} with ${threads} threads in 3s...`);
                await ns.sleep(3000);

                //for (let j = 0; j < files.length; ++j) {
                //  ns.exec(files[j], serv, threads);
                //  await ns.sleep(1000); // sleep for 1 second
                //}
                await execFiles(ns, files, serv, threads);
                ns.tprint(`${files} successfully running on ${serv}\n\n`);
                //await ns.sleep(5000);
            }
        }
    }


    
    // Wait until we acquire the "BruteSSH.exe" program
    //while (!ns.fileExists("BruteSSH.exe", "home")) {
    //    await ns.sleep(60000);
    //}
    // IF VARIANT
    if (!ns.fileExists("BruteSSH.exe", "home")) {
        ns.tprint("File BruteSSH.exe does not exist.");
        return;
    }



    //ns.tprint("Beginning main loop - 1 port 32 gb");
    // Copy our scripts onto each server that requires 1 port and 32 GB
    // to gain root access. Then use brutessh() and nuke() to gain
    // admin access and run the scripts.
    for (let i = 0; i < servers1Port.length; ++i) {
        const serv = servers1Port[i];
        // calculate max threads
        const threads = Math.floor((ns.getServerMaxRam(serv) - ns.getServerUsedRam(serv)) / ram_req);
        // bool to determine if the current server is hackable
        let pass = true;

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
          // announce this portion
          ns.tprint(`Next server: ${serv}. Beginning in 5s...`);
          await ns.sleep(5000);

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
          await access(ns, ns.getServerNumPortsRequired(serv));

          ns.tprint(`Launching scripts '${files}' on server ${serv} with ${threads} threads in 3s...`);
          await ns.sleep(3000);

          await execFiles(ns, files, serv, threads);
          ns.tprint(`${files} successfully running on ${serv}\n\n`);
          //await ns.sleep(5000);
        }
    }


    
    // Wait until we acquire the "FTPCrack.exe" program
    //while (!ns.fileExists("FTPCrack.exe", "home")) {
    //    await ns.sleep(60000);
    //}
    // IF VARIANT
    if (!ns.fileExists("FTPCrack.exe", "home")) {
        ns.tprint("File FTPCrack.exe does not exist.");
        return;
    }



    //ns.tprint("Beginning main loop - 2 port 32 gb");
    // Copy our scripts onto each server that requires 2 ports to gain
    // root access. Then use brutessh() and nuke() and ftpcrack() to
    // gain admin access and run the scripts.
    for (let i = 0; i < servers2Port.length; ++i) {
        const serv = servers1Port[i];
        // calculate max threads
        const threads = Math.floor((ns.getServerMaxRam(serv) - ns.getServerUsedRam(serv)) / ram_req);
        // bool to determine if the current server is hackable
        let pass = true;

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
          // announce this portion
          ns.tprint(`Next server: ${serv}. Beginning in 5s...`);
          await ns.sleep(5000);

          if (ns.scp(files,serv)) {
            ns.tprint(`copied files to ${serv}`);
          } else {
            ns.tprint(`scp failed to copy files ${files} to server ${serv}`);
          }
          //ns.tprint(`test: files done copied to ${serv} and will run on ${threads} threads.`);

          await access(ns, ns.getServerNumPortsRequired(serv));

          ns.tprint(`Launching scripts '${files}' on server ${serv} with ${threads} threads in 3s`);
          await ns.sleep(3000);

          await execFiles(ns, files, serv, threads);
          ns.tprint(`${files} running on ${serv}. Sleeping for 5s`);
          await ns.sleep(5000);
        }
    }

    ns.tprint("DONE -- copying/executing scripts");
}


/*
// from ChatGPT
async function copyFiles(ns, files, target) {
    return new Promise(resolve => {
        if (ns.scp(files, target)) {
            ns.tprint("copied file(s) to ", target);
            resolve(true);
        } else {
            ns.tprint(`scp failed to copy file(s) ${files} to server ${target}`);
            setTimeout(() => resolve(copyFiles(ns, files, target)), 1000); // Retry after 1 second
        }
    });
}
*/


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
            // bad thread count
            else if (threads <= 0) {
              ns.tprint(`Bad thread count (${threads}). Retrying with 1 thread.`);
              setTimeout(() => execFiles(ns, files, target, 1), 1000); // retry after 1 second using 1 thread
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



async function access(ns, num_ports) {
    return new Promise(resolve => {
        switch (num_ports) {
            case 0:
                resolve(true);

                ns.tprint(`Nuking ${serv} in 3s...`);
                //await ns.sleep(3000);
                ns.nuke(serv);

                break;
            case 1:
                resolve(true);

                ns.tprint(`BruteSSH-ing ${serv} in 3s...`);
                //await ns.sleep(3000);
                ns.brutessh(serv);
                ns.tprint(`Nuking ${serv} in 3s...`);
                //await ns.sleep(3000);
                ns.nuke(serv);

                break;
            case 2:
                resolve(true);

                ns.tprint(`FTPCrack-ing ${serv} in 3s...`);
                //await ns.sleep(3000);
                ns.ftpcrack(serv);
                ns.tprint(`BruteSSH-ing ${serv} in 3s...`);
                //await ns.sleep(3000);
                ns.brutessh(serv);
                ns.tprint(`Nuking ${serv} in 3s...`);
                //await ns.sleep(3000);
                ns.nuke(serv);

                break;
        }
    });
}