<?php
    $dsn = 'mysql:host=joecool.highpoint.edu;dbname=CSC3212_S23_wwadsworth_db';
    $username = 'wwadsworth';
    $password = '1837109';

    try {
        $db = new PDO($dsn,$username,$password);
    } catch (PDOException $e) {
        $error_message = $e -> getMessage();
        include('database_error.php');
        exit();
    }
?>
