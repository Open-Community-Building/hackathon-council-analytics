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

        component ss.wa "WebApp" {
	        include *
	        include "->element.parent==ss.wa->"
	        autolayout lr
        }

        component ss.adm "AdminCli" {
            include "->element.parent==ss.adm->"
            autolayout tb
        }

        component ss.af "Framework" {
            include "->element.parent==ss.af->"
            autolayout rl   
        }

        component ss.fs "FileStorage" {
            include "->element.parent==ss.fs->"
            autolayout bt
        }
        

       styles {
            element "Element" {
                color #ffffff
            }
            element "Group" {
                background #123456
		        fontSize 34
		        border solid
		        strokeWidth 5
                stroke orange
            }
            element "automation" {
 		        stroke yellow
		        shape Robot
                color black
            }
            element "SubCommand" {
		        shape Hexagon
	        }
            element "Web" {
                shape WebBrowser
                background lightblue
            }
            element "Person" {
                background #d34407
                shape person
            }
            element "Software System" {
                background #f86628
            }
            element "Container" {
                background #388728
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
