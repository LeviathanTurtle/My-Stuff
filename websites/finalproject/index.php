<?php 
    require('database.php');

    $willsAction = filter_input(INPUT_GET,'willsAction',FILTER_VALIDATE_INT);

    if($willsAction == NULL || $willsAction == FALSE) {
        $willsAction = 'working';
    }

    $queryAction = "SELECT * FROM actions ORDER BY RAND() LIMIT 1";
    $statement1 = $db->prepare($queryAction);
    $statement1->bindValue(':willsAction',$willsAction);
    $statement1->execute();

    $action = $statement1->fetch();
    $action_name = $action['activity'];
#    $statement1->closeCursor();



    $willsPlace = filter_input(INPUT_GET,'willsPlace',FILTER_VALIDATE_INT);

    if($willsPlace == NULL || $willsPlace == FALSE) {
        $willsPlace = 'at home';
    }

    $queryPlace = "SELECT * FROM places ORDER BY RAND() LIMIT 1";
    $statement2 = $db->prepare($queryPlace);
    $statement2->bindValue(':willsPlace',$willsPlace);
    $statement2->execute();

    $place = $statement2->fetch();
    $place_name = $place['places'];
    $statement2->closeCursor();



    $x = rand(0,7);
    if($x == 0) {
        $date = 'seconds';
    } elseif ($x == 1) {
        $date = 'minutes';
    } elseif ($x == 2) {
        $date = 'hours';
    } elseif ($x == 3) {
        $date = 'days';
    } elseif ($x == 4) {
        $date = 'weeks';
    } elseif ($x == 5) {
        $date = 'months';
    } else {
        $date = 'years';
    }

?>

<!DOCTYPE HTML>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>dead inside</title>
        <link rel="stylesheet" href="index.css">
        <!-- GOOGLE FONTS FTW -->
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
    </head>

    <header>

        <div class="flex-container">

            <!-- LOGO -->
            <div class="logo-box">
                <div class="logo-margin">
                    <a href="index.html">
                        <div class="logo-content">
                            <img src="malwartlogo.jpg" alt="logo" width="50" height="50" />
                            <div class="title">HOME</div>
                        </div>
                    </a>
                </div>
            </div>

            <!-- MIDDLE-NAV -->
            <div class="links-box">
                <div class="links-content">    
                    <a href="#About">About</a>
                    <a href="#FAQ">FAQ</a>
                    <a href="#Contact">Contact</a>
                </div>
            </div>
<!--
             SEARCH 
            <div class="search-box">
                <div class="search-content">
                    <input type="text" placeholder="Search...">
                </div>
            </div>
-->
        </div>

    </header>



    <main>
        
        <div class="main-box">
            <div class="main-content">
                <div class="main-title">
                    <h2>What is William doing now?</h2><br>
                    <!--<p>He is <?php #echo $action['activity']; ?>.</p>-->

                    <h2>When will he respond?</h2><br>
                    <!--<p>He will respond in <?php #echo $place['places']; echo $time['time']; ?></p>-->
                </div>

                <p>He is <?php echo $action_name; ?>.</p>
                <p>He will respond in <?php echo $place_name; echo " "; echo rand(0,999); echo " "; echo $date; ?>.</p>
                <br><br><br><br><br><br><br><br><br><br><br><br>

                <!-- future projects -->
                <h2 class="main-title">Projects</h2>
                <div class="main-project">
                    <div class="project-content">
                        <div class="project-box">
                            <div class="project-image">
                                <img src="comingsoon.jpg">
                            </div>

                            <div class="project-name-content">
                                <div class="project-details">
                                    <div class="project-title">[REDACTED]</div>
                                    <div class="project-description">[REDACTED]</div>
                                </div>
                            </div>

                            <div class="project-release-content">
                                <div class="project-release">
                                    <code class="releasedate">TBD</code>
                                </div>
                            </div>
                        </div>

                        <div class="project-box">
                            <div class="project-image">
                                <img src="comingsoon.jpg">
                            </div>
    
                            <div class="project-name-content">
                                <div class="project-details">
                                    <div class="project-title">[REDACTED]</div>
                                    <div class="project-description">[REDACTED]</div>
                                </div>
                            </div>
    
                            <div class="project-release-content">
                                <div class="project-release">
                                    <code class="releasedate">TBD</code>
                                </div>
                            </div>
                        </div>
    
                        <div class="project-box">
                            <div class="project-image">
                                <img src="comingsoon.jpg">
                            </div>
    
                            <div class="project-name-content">
                                <div class="project-details">
                                    <div class="project-title">[REDACTED]</div>
                                    <div class="project-description">[REDACTED]</div>
                                </div>
                            </div>
    
                            <div class="project-release-content">
                                <div class="project-release">
                                    <code class="releasedate">TBD</code>
                                </div>
                            </div>
                        </div>

                    </div>
                    
                </div>

                <br><br><br><br><br><br><br><br><br><br><br><br>
                <!-- other pages (projects) -->
                <div class="below-title">
                    <!-- link to audio page -->
                    <a href="audio.html"><h2>Audio</h2></a><br><br>

                    <!-- link to game page -->
                    <a href="games.html"><h2>Games</h2></a><br><br>

                    <!-- link to calculators page -->
                    <a href="calculators.html"><h2>Calculators</h2></a><br><br>
                </div>

            </div>
        </div>
        
    </main>



    <br><br><br><br><br><br><br><br><br><br><br><br>



    <footer>
        
        <div class="copyright-box">
            <div class="copyright">
                Copyright &copy; <script>document.write(new Date().getFullYear())</script> William Wadsworth. All Rights Reserved.<br>
            </div>
        </div>

    </footer>

</html>
