# Steps to Set Up DynamoDB on AWS
1. Create an AWS Account:

   If you don't have an AWS account, sign up at AWS. -> TO-DO

1. Log into AWS Management Console:

   Go to the AWS Console and log in.

1. Navigate to DynamoDB:

   In the AWS Console, search for DynamoDB in the search bar and click on the service.

1. Create a DynamoDB Table:

   In the DynamoDB dashboard, click on Create Table.
   - **Table Name**: Enter a name for your table (e.g., Users).
   - **Primary Key**:
     - Choose a Partition Key (e.g., UserID).
     - Optionally, add a Sort Key if you're creating a composite key (e.g., Email).
   - **Read/Write Capacity**: You can choose between Provisioned (manual scaling) and On-Demand (automatic scaling based on usage). For small apps, On-Demand is easier to manage.
   
   Once you’ve configured these settings, click **Create Table**.

1. Enable Access to DynamoDB (IAM Role Setup):

   Ensure that your AWS user has the required permissions to interact with DynamoDB.
   Create or update an IAM role/policy to give it DynamoDB access by attaching the **AmazonDynamoDBFullAccess** policy.
