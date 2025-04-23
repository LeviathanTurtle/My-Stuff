# A Baldur's Gate 3 Exploit Macro

Note to clear up any confusion: this directory is a collection of scripts I wrote for use when playing [Baldur's Gate 3](https://store.steampowered.com/app/1086940/Baldurs_Gate_3/). This should NOT be confused with the other Games directory, which is a collection of games to play.

## Background
In Baldur's Gate 3, certain classes are granted access to spell slots. These are used as a sort of currency and allow the character to cast certain learned spells. For example, Level 1 Sorcerers are granted two first-level spell slots, meaning they can cast two first-level spells. All spell slots are only refreshed after a long rest (ending the day), so as long as you consistently have enough materials to do so, you can just repeatedly do that after running out of spell slots, right?

Well, a potential problem with this is in the highest difficulty (Honor Mode), consecutive long rests get progressively more expensive, not to mention any buffs your party has will go away and any time sensitive quests may be affected after ending the day enough times.

## Context
One of the things that makes Sorcerers unique is that they have access to what's called Sorcery Points (also refreshed after a long rest), and they can be used to activate certain class [Metamagic](https://bg3.wiki/wiki/Metamagic) or create spell slots.

This macro utilizes an infinite sorcery points and spell slots [exploit](https://youtu.be/McnZwKkqanQ?si=U0h2VT1BLgwUji96) (Note: the video title is clickbait; it is SFW) to get an infinite amount of spell slots. Certain gear comes with bonuses in the form of stat boosts (applied when equipped) or skills (usually refreshed after a long rest). A select few (specifically two, I believe) give the player an additional spell slot. The basic premise of the exploit is to equip this piece of gear, use the spell slot, unequip, and repeat. For some reason, the spell slot benefit is still able to be used. Normally this is not the case; if you use a skill from a piece of gear, unequipping and re-equipping will not allow you to use the skill again until you end the day via a long rest.

Below are before and after images for a Level 12 Sorcerer.

![Base](./base.jpg)
![After](./after.jpg)
*<br>Note that in Vanilla, the maximum spell level you can convert is level 5*

It seems that the game stores the maximum number of spell slots as a number, which is inflated by the exploit. I think the reason long-resting does not work in the same manner as Angelic Reprieve potions (which restores health and first and second level spell slots) is that long-resting also resets any party buffs and will hard reset this value to what it is supposed to be (as defined in your class tree(s)).

## What this does
The macro utilizes this exploit (the second, third, and fourth methods outlined in the video) to give you the amount of spell slots and sorcery points you specify. The fastest method is to use freecast, but that is not available until Act 3, as is the amulet. <b>The Freecast version was also fixed in Patch 8.</b> The shield is available from a vendor in Act 2, so that is the earliest this macro can be used. Please note that you can only use the amulet with this macro if you are using a version before Patch 7 (I do not know which specific version). The shield and freecast methods both work as of Patch 7. As of now, this only works on Windows.

## Disclaimers
Note that icon positions and subsequent mouse positions in the macro are the measurements I made, so they may not be the same for you. I recommend testing the mouse positions first with `test_coords.py` to see if my measurements match yours. If not, or if you would like to adjust the U.I. to your liking, you can use `mouse_pos.py` in Tools to get the new coordinates. `test_coords.py` will tell you where to update the values in the macro. Also be sure to adjust the spell slot and sorcerer point amounts to your liking before using. In my testing there has been no accidental icon moving by the macro, but just in case, I recommend locking the U.I. 

Here is how the U.I. should be organized if you do not want to do any re-arranging:
![U.I.](./ui.jpg)

This macro requires being a Sorcerer of at least level 2. This works with a minimum of 30 FPS (but higher will be more consistent; I like ~45) at 1920x1080 resolution and should work at any point in the game. If are using the shield or amulet and the delta between your current and target spell slots is less than your current number of slots, the macro will give you extra first or second level spell slots. This is because the macro assumes you will have none of the spell slot level the gear gives you, and because the game will use your base class spell slots before the slot given by the gear. 

## Execution
1. Check the coordinates of the icons
2. Update the marked values at the top of the script to what you want
3. Unequip any two-handed weapon, if you are using one
4. Open a terminal window and navigate to the script
5. Run `python bg3_inf_sorcspell.py`
6. Ensure the Baldur's Gate 3 window is in focus
7. Activate using the keybind set in Step 2

You should see a message in the terminal and the macro will begin shortly. Please note that the macro is so powerful, there is no graceful way of stopping it, so ensure that the Baldur's Gate 3 window remains in focus until the macro is finished. <b>This is not a fast macro if you are using the shield or amulet and want a lot of spell slots,</b> so do not expect it to be done in a minute or two. It took the following times to create the earlier before/after example:

| shield | freecast |
| ------ | -------- |
| 1917.27s (31.95 min) | 375.23s (6.25 min)

