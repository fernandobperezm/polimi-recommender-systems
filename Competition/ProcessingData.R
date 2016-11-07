# Politecnico di Milano
# ProcessingData.r
#
# Description: This script contains the R Code of Algorithm A for RecSys 2016
#              competition.
#
# Created by: Fernando Pérez on 19/10/2016.
#
# Last Modified: 19/10/2016.

# Getting interactions.
inter <- read.delim(
    "~/Development/usb-projects/polimi-projects/Recommender_Systems/Competition/interactions.csv"
    )


# Getting the user profile.
up <- read.delim(
    "~/Development/usb-projects/polimi-projects/Recommender_Systems/Competition/user_profile.csv",
    na.strings = c("","NA","NULL")
    )

# Getting target users.
tu <- read.delim(
    "~/Development/usb-projects/polimi-projects/Recommender_Systems/Competition/originaldata/target_users.csv"
    )

# Getting item's profiles.
ip <- read.delim(
    "~/Development/usb-projects/polimi-projects/Recommender_Systems/Competition//originaldata/item_profile.csv",
    na.strings = c("","NA","NULL")
    )

# write.table(ip, "ip.csv", quote = F, sep="!", na = "nan",row.names=F)


