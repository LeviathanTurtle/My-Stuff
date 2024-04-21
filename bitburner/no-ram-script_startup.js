/** @param {NS} ns */

// 5.75 GB total RAM across 3 scripts
// 16 GB : 2 threads each (4.5 free)
// 32 GB : 5 threads each (3.25 free)
// 64 GB : 11 threads each (.75 free)
// 128 GB : 22 threads each (1.5 free)
// 256 GB : ...

// Several servers have 0 GB RAM available. This means that scripts cannot be run
// on those servers. Instead, they must be running on a separate server. This
// script purchases a new private server, uploads the main 3 attacking scripts,
// and executes them on said server attacking one of the 0 GB RAM servers. This is
// repeated until we cannot buy more private servers, or until there are no more
// servers to hack. 

export async function main(ns, ram) {
    // array of main files to copy and use
    const files = ["weaken-template.js", "hack-template.js", "grow-template.js"];
    //ns.tprint(`TEST: ns.getHostName = ${ns.getHostname()}`);

    // calculate total script ram usage
    let ram_req = 0;
    for (let i=0; i < files.length; ++i) {
      ram_req += ns.getScriptRam(files[i]);
    }
    //ns.tprint(`total script ram required: ${ram_req}\n\n`);

    // How much RAM each purchased server will have
    //const ram = 64;

    // first collection has 2 ports
    const servers_no_ram = ["johnson-ortho",
                            "crush-fitness",
                            "zb-def",
                            "nova-med",
                            "syscore",
                            "snap-fitness",
    // 3 ports
                            "computek",
    // 5 ports
                            "galactic-cyber",
                            "aerocorp",
                            "defcomm",
                            "icarus",
                            "infocomm",
                            "taiyang-digital",
                            "deltaone",
                            "zeus-med"                      
    ];


    // Sort `servers_no_ram` from lowest hacking level to highest
    servers_no_ram.sort((a, b) => {
        // Get the required hacking level for each server
        const hackingLevelA = ns.getServerRequiredHackingLevel(a);
        const hackingLevelB = ns.getServerRequiredHackingLevel(b);

        // Return the difference to sort the array
        return hackingLevelA - hackingLevelB;
    });


    // server loop array loop control
    let j = 0;
    // Continuously try to purchase servers until we've reached the maximum
    // amount of servers
    for(let i=0; i < ns.getPurchasedServerLimit(); ++i) {
        const serv = servers_no_ram[j];
        // bool to determine if the current server is hackable
        let hackable = true;


        // check for current hack level vs. server
        if (ns.getHackingLevel() < ns.getServerRequiredHackingLevel(serv)) {
          ns.tprint(`Current server ${serv} is not currently hackable\n\n`);
          hackable = false;
        }

        
        // we can hack the server
        if (hackable) {
          // Check if we have enough money to purchase a server
          if (ns.getServerMoneyAvailable("home") > ns.getPurchasedServerCost(ram)) {
              // 1. announce purchase, purchase server 
              ns.tprint(`Purchasing ${ram} GB server to attack ${serv} in 1s...`);
              await ns.sleep(1000);
              let hostname = ns.purchaseServer(`pserv-${i}-${servers_no_ram[i]}`, ram);

              // 2. copy files to new server
              if (ns.scp(files,hostname)) {
                ns.tprint(`copied files to ${hostname}`);
              } else {
                ns.tprint(`scp failed to copy files ${files} to server ${hostname}`);
              }
              //ns.tprint(`test: files done copied to ${serv} and will run on ${threads} threads.`);
              
              // 3. execute scripts with max threads
              let threads = Math.floor((ns.getServerMaxRam(hostname) - ns.getServerUsedRam(hostname)) / ram_req);
              if (threads > 0) {
                  ns.tprint(`Launching scripts '${files}' on ${hostname} with ${threads} threads in 1s...`);
                  await ns.sleep(1000);
                  await execFiles(ns, files, hostname, threads);

                  ns.tprint(`All files successfully running on ${hostname}\n\n`);
              } else {
                ns.tprint(`Files not running on ${hostname}\n\n`);
              }
          }
        }

        // increment for next server
        j++;
        
        // Make the script wait for a second before looping again.
        // Removing this line will cause an infinite loop and crash the game.
        await ns.sleep(1000);
    }

    ns.tprint("DONE -- purchasing private servers to attack low-memory servers");
    return;
}



async function execFiles(ns, files, serv, target, threads) {
    return new Promise(resolve => {
        const executeFile = async (fileIndex) => {
            if (fileIndex >= files.length) {
                resolve(true); // all files executed successfully
                return;
            }
            
            const file = files[fileIndex];
            // successful start
            //if (ns.exec(file, "home", threads, ...target)) {
            if (ns.run(file,threads,target)) {
                ns.tprint(`File ${file} running on ${serv}`);
                setTimeout(() => executeFile(fileIndex + 1), 1000); // execute next file after 1 second
            }
            // could not execute file
            else {
                ns.tprint(`Failed to start file ${file} on ${serv}`);
                setTimeout(() => executeFile(fileIndex), 1000); // retry current file after 1 second
            }
        };

        executeFile(0); // start executing files from the beginning of the array
    });
}

