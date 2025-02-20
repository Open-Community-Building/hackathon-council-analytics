u1 = person "User"
u2 = person "Admin"
u3 = person "Power User"
#u4 = person "Kunden"
r1 = person "Cronjob" {
    tags "automation"
}
ss = softwareSystem "Council Insights" {
    wa = container "Web Application" {
        cb = component "Chatbot"
        cg = component "Config GUI"
    }
    af = container "AI Framework" {
        ai = component "AI Model"
        vd = component "Vector Database" {
            tags "Database"
        }
    }
    adm = container "Admin CLI" {
        dl = component "Download" {
            tags "SubCommand"
        }
        pp = component "PreProcessor" {
            tags "SubCommand"
        }
        emb = component "Embedor" {
            tags "SubCommand"
        }
        upd = component "Updater" {
            tags "SubCommand"
        }
    }
    fs = container "FileStorage" {
        tags "Database"
        nc = component "Nextcloud"
        fsys = component "Filesystem"
    }
    src = container "RAG Source" {
        tags "external" "Web"
    }
                    
}

        u1 -> ss.wa.cb "Uses"
        u2 -> ss.adm.upd "Uses"
        u3 -> ss.wa.cg "Uses"
        #u4 -> ss.wa "Uses"
        r1 -> ss.adm.upd "starts"
        ss.wa.cb -> ss.af.ai "queries"
        ss.adm.upd -> ss.af.vd "indexes"
        ss.af.ai -> ss.af.vd "reads"
        ss.adm.upd -> ss.adm.emb "calls"
        ss.adm.emb -> ss.adm.pp "calls"
        ss.adm.pp -> ss.adm.dl "calls"
        ss.adm.pp -> ss.fs "writes"
        ss.adm.pp -> ss.fs.nc "remote access"
        ss.adm.pp -> ss.fs.fsys "local read/write"
        ss.adm.emb -> ss.fs "reads"
        ss.adm.dl -> ss.src "pulls"
        ss.fs.nc -> ss.fs.fsys "read and writes"
        
