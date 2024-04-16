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

    // Array of all servers
    const servers = ["foodnstuff",       // 16 GB
                     "sigma-cosmetics",  // 16 GB
                     "joesguns",         // 16 GB
                     "nectar-net",       // 16 GB
                     "hong-fang-tea",    // 16 GB
                     "harakiri-sushi",   // 16 GB
                     "max-hardware",     // 32 GB
                     "neo-net",          // 32 GB
                     "zer0",             // 32 GB
                     "iron-gym",         // 32 GB
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

    // calculate total script ram usage
    let ram_req = -4;
    for (let i=0; i < files.length; ++i) {
      ram_req += ns.getScriptRam(files[i]);
    }
    //ns.tprint(`total script ram required: ${ram_req}\n\n`);
    
    // variable to see how many servers are affected
    let affect_server_count = 0;

    let threads = Math.floor((ns.getServerMaxRam("home") - ns.getServerUsedRam("home")) / ram_req / servers.length);
    if (threads <= 0) {
      threads = 1;
    }



    // Just run the scripts from home
    for (let i = 0; i < servers.length; ++i) {
        const serv = servers[i];
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

        if (pass) {
          ns.tprint(`Launching scripts '${files}' on home with ${threads} threads in 1s...`);
          await ns.sleep(1000);

          await execFiles(ns, files, serv, threads);

          ns.tprint(`All files successfully running on ${serv}\n\n`);
          //await ns.sleep(5000);
          affect_server_count++;
        }
    }

    ns.tprint("DONE -- executing scripts on home");
    ns.tprint(`Affected servers: ${affect_server_count}`);
    return;
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
            //if (ns.exec(file, "home", threads, ...target)) {
            if (ns.run(file,threads,target)) {
                ns.tprint(`File ${file} running on home`);
                setTimeout(() => executeFile(fileIndex + 1), 1000); // Execute next file after 1 second
            }
            // could not execute file
            else {
                ns.tprint(`Failed to start file ${file} on home`);
                setTimeout(() => executeFile(fileIndex), 1000); // Retry current file after 1 second
            }
        };

        executeFile(0); // Start executing files from the beginning of the array
    });
}

