workspace "Name" "Description" {

    !identifiers hierarchical

    model {
        u1 = person "User"
        u2 = person "Admin"
        u3 = person "Gäste"
        u4 = person "Kunden"
        ss = softwareSystem "Software System" {
            wa = container "Web Application"
            db = container "Database Schema" {
                tags "Database"
            }
        }

        u1 -> ss.wa "Uses"
        u2 -> ss.wa "Uses"
        u3 -> ss.wa "Uses"
        u4 -> ss.wa "Uses"
        ss.wa -> ss.db "Reads from and writes to"
    }

    views {
        systemContext ss "Diagram1" {
            include *
            autolayout lr
        }

        container ss "Diagram2" {
            include *
            autolayout lr
        }

        styles {
            element "Element" {
                color #ffffff
            }
            element "Person" {
                background #d34407
                shape person
            }
            element "Software System" {
                background #f86628
            }
            element "Container" {
                background #f88728
            }
            element "Database" {
                shape cylinder
            }
        }
    }

    configuration {
        scope softwaresystem
    }

}
