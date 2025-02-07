workspace "Council Insights" "General Documentation of Project and Infrastructure" {

    !identifiers hierarchical
    !adrs adr

    model {
        !include model.dsl
    }

    views {
        systemContext ss "C1_Context" {
            include *
            autolayout lr
        }

        container ss "Container" {
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
