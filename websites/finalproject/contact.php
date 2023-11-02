<?php
    // get data from form
#    $fname = filter_input(INPUT_POST,'fname',FILTER_VALIDATE_INT);
    $fname = $_POST['fname'];
#    $lname = filter_input(INPUT_POST,'lname',FILTER_VALIDATE_INT);
    $lname = $_POST['lname'];
#    $email = filter_input(INPUT_POST,'email',FILTER_VALIDATE_INT);
    $email = $_POST['email'];

    if($fname == null || $fname == FALSE || $lname == null || $lname == FALSE || $email == null || $email == FALSE) {
        echo "invalid data, please try again.";
#        $error = "invalid data, please try again.";
#        include('error.php');
    } else {
        require_once('database.php');
    }

    $query = "INSERT INTO people (peopleFName, peopleLName, peopleEmail) VALUES (:fname, :lname, :email)";
    $statement = $db->prepare();
    $statement->bindValue(':fname',$fname);
    $statement->bindValue(':lname',$lname);
    $statement->bindValue(':email',$email);
    $statement->execute();
    $statement->closeCursor();
?>



<!DOCTYPE html>
<html>

    <head>
        <meta charset="utf-8">
        <title>Thanks</title>
        <link rel="stylesheet" href="index.css">
        <!-- GOOGLE FONTS FTW -->
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">

        <style>

            html {
                font-family: 'Roboto', sans-serif;
                background-color: #1A1B1E;
                color: #c1c2c5;
            }

        </style>
    </head>

    <header>

        <div class="flex-container">

            <!-- LOGO -->
            <div class="logo-box">
                <div class="logo-margin">
                    <a href="index.html">
                        <div class="logo-content">
                            <img src="malwartlogo.jpg" alt="logo" width="50" height="50">
                            <div class="title">HOME</div>
                        </div>
                    </a>
                </div>
            </div>

        </div>

    </header>


    <body>
        <main>
            <p>Thanks! We will email you when your account has been created</p><br>
            <a href="index.html">Return home</a>
        </main>
    </body>

    <footer>
        
        <div class="copyright-box">
            <div class="copyright">
                Copyright &copy; <script>document.write(new Date().getFullYear())</script> William Wadsworth. All Rights Reserved.<br>
            </div>
        </div>

    </footer>


</html>
