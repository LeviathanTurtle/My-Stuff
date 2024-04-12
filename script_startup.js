/** @param {NS} ns */

//ns.getServerMaxRam()
//ns.getFunctionRamCost()
//ns.getHostname()

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
    // to gain root access. These have 4 GB of RAM
    const servers0Port4gb = ["n00dles"];

    // Array of all servers that don't need any ports opened
    // to gain root access. These have 16 GB of RAM
    const servers0Port16gb = ["foodnstuff",
                              "sigma-cosmetics",
                              "joesguns",
//                              "nectar-net",
                              "hong-fang-tea",
                              "harakiri-sushi"];
/*
    // Array of all servers that only need 1 port opened
    // to gain root access. These have 32 GB of RAM
    const servers1Port32gb = [//"max-hardware",
//                              "neo-net",
//                              "zer0",
                              "iron-gym"];
    
    // Array of all servers that only need 2 ports opened
    // to gain root access. These have 32 GB of RAM
    const servers2Port32gb = ["phantasy",
                              "omega-net"];
    
    // Array of all servers that only need 2 ports opened
    // to gain root access. These have 64 GB of RAM
    const servers2Port64gb = ["silver-helix"];

    // Array of all servers that only need 2 ports opened
    // to gain root access. These have 128 GB of RAM
    const servers2Port128gb = ["avmnite-02h"];
*/
    ns.tprint("Beginning main loop - 0 port 4 gb");
    // Copy our scripts onto each server that requires 0 ports and 4 GB
    // to gain root access. Then use nuke() to gain admin access and
    // run the scripts.
    for (let i = 0; i < servers0Port4gb.length; ++i) {
        const serv = servers0Port4gb[i];

        //ns.scp("early-hack-template.js", serv);
        //if (ns.scp("early-hack-template.js", serv)) {
        //  ns.tprint("did that thang 0-4");
        //}
        await copyFiles(ns, "early-hack-template.js", serv)

        //ns.tprint("test-servers0Port4gb");
        ns.nuke(serv);
        //ns.tprint("end-test-servers0Port4gb\n");

        //await ns.sleep(1000); // sleep for 1 second
        //await delay(1000); // wait for 1 second
        ns.tprint(`Launching script(s) 'early-hack-template.js' on server '${serv}' with 1 thread`);
        ns.exec("early-hack-template.js", serv);
    }

    ns.tprint("Sleeping...");
    await ns.sleep(5000);
    ns.tprint("Beginning main loop - 0 port 16 gb");
    // Copy our scripts onto each server that requires 0 ports and 16 GB
    // to gain root access. Then use nuke() to gain admin access and
    // run the scripts.
    for (let i = 0; i < servers0Port16gb.length; ++i) {
        const serv = servers0Port16gb[i];
        // calculate max threads
        const threads = Math.floor((ns.getServerMaxRam(serv) - ns.getServerUsedRam(serv)) / ram_req);

        //ns.scp("weaken-template.js", serv);
        //ns.scp("hack-template.js", serv);
        //ns.scp("grow-template.js", serv);
        //ns.tprint("about to do that thang 0-16");
        //if (ns.scp(files, serv)) {
        //  ns.tprint("did that thang 0-16");
        //}
        await copyFiles(ns, files, serv)
        //ns.tprint(`test: files done copied to ${serv} and will run on ${threads} threads.`);

        //ns.tprint("test-servers0Port16gb");
        ns.tprint(`Nuking ${serv} in 5 seconds...`);
        await ns.sleep(5000);
        ns.nuke(serv);
        //ns.tprint("end-test-servers0Port16gb\n");

        //await ns.sleep(1000); // sleep for 1 second
        //await delay(1000); // wait for 1 second
        //ns.exec("weaken-template.js", serv, 2);
        //await ns.sleep(1000); // sleep for 1 second
        //await delay(1000); // wait for 1 second
        //ns.exec("hack-template.js", serv, 2);
        //await ns.sleep(1000); // sleep for 1 second
        //await delay(1000); // wait for 1 second
        //ns.exec("grow-template.js", serv, 2);
        ns.tprint(`Launching script(s) '${files}' on server '${serv}' with ${threads} threads`);
        for (let j = 0; j < files.length; ++j) {
          ns.exec(files[j], serv, threads);
          await ns.sleep(1000); // sleep for 1 second
        }
    }

    /*
    // Wait until we acquire the "BruteSSH.exe" program
    //while (!ns.fileExists("BruteSSH.exe", "home")) {
    //    await ns.sleep(60000);
    //}

    ns.tprint("Sleeping...");
    await ns.sleep(5000);
    ns.tprint("Beginning main loop - 1 port 32 gb");
    // Copy our scripts onto each server that requires 1 port and 32 GB
    // to gain root access. Then use brutessh() and nuke()
    // to gain admin access and run the scripts.
    for (let i = 0; i < servers1Port32gb.length; ++i) {
        const serv = servers1Port32gb[i];

        //ns.scp("weaken-template.js", serv);
        //ns.scp("hack-template.js", serv);
        //ns.scp("grow-template.js", serv);
        ns.scp(files, serv);

        ns.tprint("test-servers1Port32gb");
        ns.brutessh(serv);
        ns.nuke(serv);
        //ns.exec("backdoor", serv);
        ns.tprint("end-test-servers1Port32gb\n");

        await ns.sleep(1000); // sleep for 1 second
        ns.exec("weaken-template.js", serv, 5);
        await ns.sleep(1000); // sleep for 1 second
        ns.exec("hack-template.js", serv, 5);
        await ns.sleep(1000); // sleep for 1 second
        ns.exec("grow-template.js", serv, 5);
    }
    
    // Wait until we acquire the "FTPCrack.exe" program
    while (!ns.fileExists("FTPCrack.exe", "home")) {
        await ns.sleep(60000);
    }

    ns.tprint("Sleeping...");
    await ns.sleep(5000);
    ns.tprint("Beginning main loop - 2 port 32 gb");
    // Copy our scripts onto each server that requires 2 ports and 32 GB
    // to gain root access. Then use brutessh() and nuke() and ftpcrack()
    // to gain admin access and run the scripts.
    for (let i = 0; i < servers2Port32gb.length; ++i) {
        const serv = servers2Port32gb[i];

        //ns.scp("weaken-template.js", serv);
        //ns.scp("hack-template.js", serv);
        //ns.scp("grow-template.js", serv);
        ns.scp(files, serv);

        ns.tprint("test-servers2Port32gb");
        ns.brutessh(serv);
        ns.ftpcrack(serv);
        ns.nuke(serv);
        //ns.exec("backdoor", serv);
        ns.tprint("end-test-servers2Port32gb\n");

        await ns.sleep(1000); // sleep for 1 second
        ns.exec("weaken-template.js", serv, 5);
        await ns.sleep(1000); // sleep for 1 second
        ns.exec("hack-template.js", serv, 5);
        await ns.sleep(1000); // sleep for 1 second
        ns.exec("grow-template.js", serv, 5);
    }

    ns.tprint("Sleeping...");
    await ns.sleep(5000);
    ns.tprint("Beginning main loop - 2 port 64 gb");
    // Copy our scripts onto each server that requires 2 ports and 64 GB
    // to gain root access. Then use brutessh() and nuke() and ftpcrack()
    // to gain admin access and run the scripts.
    for (let i = 0; i < servers2Port64gb.length; ++i) {
        const serv = servers2Port64gb[i];

        //ns.scp("weaken-template.js", serv);
        //ns.scp("hack-template.js", serv);
        //ns.scp("grow-template.js", serv);
        ns.scp(files, serv);

        ns.tprint("test-servers2Port64gb");
        ns.brutessh(serv);
        ns.ftpcrack(serv);
        ns.nuke(serv);
        //ns.exec("backdoor", serv);
        ns.tprint("end-test-servers2Port32gb\n");

        await ns.sleep(1000); // sleep for 1 second
        ns.exec("weaken-template.js", serv, 11);
        await ns.sleep(1000); // sleep for 1 second
        ns.exec("hack-template.js", serv, 11);
        await ns.sleep(1000); // sleep for 1 second
        ns.exec("grow-template.js", serv, 11);
    }

    ns.tprint("Sleeping...");
    await ns.sleep(5000);
    ns.tprint("Beginning main loop - 2 port 128 gb");
    // Copy our scripts onto each server that requires 2 ports and 128 GB
    // to gain root access. Then use brutessh() and nuke() and ftpcrack()
    // to gain admin access and run the scripts.
    for (let i = 0; i < servers2Port128gb.length; ++i) {
        const serv = servers2Port128gb[i];

        //ns.scp("weaken-template.js", serv);
        //ns.scp("hack-template.js", serv);
        //ns.scp("grow-template.js", serv);
        ns.scp(files, serv);

        ns.tprint("test-servers2Port128gb");
        ns.brutessh(serv);
        ns.ftpcrack(serv);
        ns.nuke(serv);
        //ns.exec("backdoor", serv);
        ns.tprint("end-test-servers2Port128gb\n");

        await ns.sleep(1000); // sleep for 1 second
        ns.exec("weaken-template.js", serv, 22);
        await ns.sleep(1000); // sleep for 1 second
        ns.exec("hack-template.js", serv, 22);
        await ns.sleep(1000); // sleep for 1 second
        ns.exec("grow-template.js", serv, 22);
    }

    */
    ns.tprint("DONE.");
}

// from ChatGPT
//function delay(ms) {
//  return new Promise(resolve => setTimeout(resolve, ms));
//}

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

