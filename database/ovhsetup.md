# Setup of MongoDB in OVH
1. Create a Public Cloud Project

    Ensure you have access to the OVHcloud Control Panel and can create a new project under the Public Cloud section.

1. Generate SSH Keys 
    
    Confirm that you can generate SSH keys on your local machine using the command:
    ```bash
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    ```
    Check: Make sure the public key is added to the OVHcloud Control Panel under Public Cloud > SSH Keys.

1. Create a Public Cloud Instance (Note: I researched all these options on the internet so it is possible that the user interface might be slightly different)
    - Database: MongoDB 4.4
    - Database Solution: Essential
    - Database Name: HeidelbergCounsel
    - Region: Select a region close to your users, such as France or Germany.
    - Image: Choose Ubuntu for its user-friendliness and wide support.
    - SSH Key: Assign the SSH key you generated.
    - Network: Default settings are usually fine.
    
1. Install NoSQL Database (e.g., MongoDB)
    
    Connect to your instance via SSH:
    ```bash
    ssh root@your_instance_ip
    ```
    Install MongoDB:
    ```bash
    sudo apt update
    sudo apt install -y mongodb
    ```
    Start and Enable MongoDB:
    ```bash
    sudo systemctl start mongodb
    sudo systemctl enable mongodb
    ```

5. Configure MongoDB
    
    Edit the MongoDB configuration file (/etc/mongodb.conf) to bind to your instance’s IP address.
    Restart MongoDB:
    ```bash
    sudo systemctl restart mongodb
    ```
6. Verify the Installation
    Check MongoDB Status:
    ```bash
    sudo systemctl status mongodb
    ```