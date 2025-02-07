u1 = person "User"
u2 = person "Admin"
u3 = person "Gäste"
u4 = person "Kunden"
        ss = softwareSystem "Council Insights" {
            wa = container "Web Application"
            cb = container "Chatbot"
            db = container "File Storage" {
                tags "Database"
                technology "Nextcloud"
            }
        }

        u1 -> ss.wa "Uses"
        u2 -> ss.wa "Uses"
        u3 -> ss.wa "Uses"
        u4 -> ss.wa "Uses"
        ss.wa -> ss.cb
        ss.wa -> ss.db "Reads from and writes to"

