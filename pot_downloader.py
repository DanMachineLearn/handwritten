# -*- encoding: utf-8 -*-
'''
@File		:	pot_downloader.py
@Time		:	2024/05/09 09:21:08
@Author		:	dan
@Description:	专门负责下载线上pot文件的类
'''


import os
from alive_progress import alive_bar
import my_wget


class PotDownloader :
    '''
    专门下载pot文件的下载器
    '''

    def __init__(self, download_url : str = 'https://gonsin-common.oss-cn-shenzhen.aliyuncs.com/handwritten/', base_dir = "work") -> None:
        ''' 
        
        Parameters
        ----------
        download_url 基本的下载地址
        
        base_dir 基本的本地地址
        
        '''
        self.__base_dir = base_dir
        self.__download_url = download_url
        # 遍历 pot 文件数量
        file_count = 0
        file_list = []
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        file_list = self.get_filelist(base_dir, file_list, '.pot')
        self.__file_count = len(file_list)


    def get_filelist(self, dir, file_list : list, ext : str = None):
        '''
        递归遍历文件夹中所有文件
        dir 需要遍历的文件夹
        
        file_list : list  文件列表，用于递归
        
        ext : str  文件后缀
        '''
        new_dir = dir
        if os.path.isfile(dir):
            if ext is None:
                file_list.append(dir)
            else:
                if dir.endswith(ext):   
                    file_list.append(dir)
            # # 若只是要返回文件文，使用这个
            # Filelist.append(os.path.basename(dir))
        elif os.path.isdir(dir):
            for s in os.listdir(dir):
                # 如果需要忽略某些文件夹，使用以下代码
                #if s == "xxx":
                    #continue
                new_dir = os.path.join(dir,s)
                self.get_filelist(new_dir, file_list)
        return file_list

    def start(self):
        ''' 
        下载放在阿里云上面的pot文件，方便随时训练
        
        Parameters
        ----------
        download_url : str 文件下载的基准网址
        '''

        if self.__file_count == 0:
            return;

        FILE_LIST = {
            "PotTrain": ["001.pot","002.pot","003.pot","004.pot","005.pot","007.pot","008.pot","009.pot","010.pot","011.pot","012.pot","014.pot","015.pot","016.pot","017.pot","018.pot","019.pot","021.pot","022.pot","023.pot","024.pot","025.pot","026.pot","027.pot","028.pot","030.pot","031.pot","032.pot","033.pot","034.pot","035.pot","036.pot","037.pot","039.pot","040.pot","041.pot","043.pot","044.pot","045.pot","047.pot","048.pot","049.pot","050.pot","051.pot","052.pot","053.pot","054.pot","055.pot","056.pot","057.pot","059.pot","061.pot","063.pot","064.pot","065.pot","066.pot","067.pot","068.pot","070.pot","071.pot","072.pot","074.pot","075.pot","076.pot","077.pot","078.pot","079.pot","080.pot","081.pot","083.pot","085.pot","086.pot","087.pot","088.pot","089.pot","090.pot","091.pot","092.pot","093.pot","094.pot","095.pot","096.pot","098.pot","099.pot","100.pot","1001.pot","1002.pot","1003.pot","1004.pot","1005.pot","1006.pot","1007.pot","1008.pot","1009.pot","101.pot","1010.pot","1011.pot","1012.pot","1013.pot","1014.pot","1015.pot","1016.pot","1017.pot","1018.pot","1019.pot","102.pot","1020.pot","1021.pot","1022.pot","1023.pot","1024.pot","1025.pot","1026.pot","1027.pot","1028.pot","1029.pot","103.pot","1030.pot","1031.pot","1032.pot","1033.pot","1034.pot","1035.pot","1036.pot","1037.pot","1038.pot","1039.pot","1040.pot","1041.pot","1042.pot","1043.pot","1044.pot","1045.pot","1046.pot","1047.pot","1048.pot","1049.pot","105.pot","1050.pot","1051.pot","1052.pot","1053.pot","1054.pot","1055.pot","1056.pot","1057.pot","1058.pot","1059.pot","106.pot","1060.pot","1061.pot","1062.pot","1063.pot","1064.pot","1065.pot","1066.pot","1067.pot","1068.pot","1069.pot","107.pot","1070.pot","1071.pot","1072.pot","1073.pot","1074.pot","1075.pot","1076.pot","1077.pot","1078.pot","1079.pot","1080.pot","1081.pot","1082.pot","1083.pot","1084.pot","1085.pot","1086.pot","1087.pot","1088.pot","1089.pot","109.pot","1090.pot","1091.pot","1092.pot","1093.pot","1094.pot","1095.pot","1096.pot","1097.pot","1098.pot","1099.pot","110.pot","1100.pot","1101.pot","1102.pot","1103.pot","1104.pot","1105.pot","1106.pot","1107.pot","1108.pot","1109.pot","1110.pot","1111.pot","1112.pot","1113.pot","1114.pot","1115.pot","1116.pot","1117.pot","1118.pot","1119.pot","112.pot","1120.pot","1121.pot","1122.pot","1123.pot","1124.pot","1125.pot","1126.pot","1127.pot","1128.pot","1129.pot","1130.pot","1131.pot","1132.pot","1133.pot","1134.pot","1135.pot","1136.pot","1137.pot","1138.pot","1139.pot","114.pot","1140.pot","1141.pot","1142.pot","1143.pot","1144.pot","1145.pot","1146.pot","1147.pot","1148.pot","1149.pot","115.pot","1150.pot","1151.pot","1152.pot","1153.pot","1154.pot","1155.pot","1156.pot","1157.pot","1158.pot","1159.pot","116.pot","1160.pot","1161.pot","1162.pot","1163.pot","1164.pot","1165.pot","1166.pot","1167.pot","1168.pot","1169.pot","117.pot","1170.pot","1171.pot","1172.pot","1173.pot","1174.pot","1175.pot","1176.pot","1177.pot","1178.pot","1179.pot","118.pot","1180.pot","1181.pot","1182.pot","1183.pot","1184.pot","1185.pot","1186.pot","1187.pot","1188.pot","1189.pot","119.pot","1190.pot","1191.pot","1192.pot","1193.pot","1194.pot","1195.pot","1196.pot","1197.pot","1198.pot","1199.pot","120.pot","1200.pot","1201.pot","1202.pot","1203.pot","1204.pot","1205.pot","1206.pot","1207.pot","1208.pot","1209.pot","1210.pot","1211.pot","1212.pot","1213.pot","1214.pot","1215.pot","1216.pot","1217.pot","1218.pot","1219.pot","122.pot","1220.pot","1221.pot","1222.pot","1223.pot","1224.pot","1225.pot","1226.pot","1227.pot","1228.pot","1229.pot","123.pot","1230.pot","1231.pot","1232.pot","1233.pot","1234.pot","1235.pot","1236.pot","1237.pot","1238.pot","1239.pot","1240.pot","126.pot","127.pot","128.pot","129.pot","130.pot","131.pot","133.pot","134.pot","135.pot","137.pot","138.pot","139.pot","140.pot","141.pot","142.pot","143.pot","144.pot","145.pot","146.pot","148.pot","149.pot","151.pot","152.pot","154.pot","155.pot","159.pot","161.pot","162.pot","163.pot","164.pot","165.pot","166.pot","167.pot","170.pot","171.pot","174.pot","175.pot","176.pot","177.pot","178.pot","179.pot","180.pot","182.pot","183.pot","184.pot","185.pot","186.pot","188.pot","190.pot","191.pot","192.pot","196.pot","197.pot","198.pot","199.pot","200.pot","201.pot","203.pot","204.pot","206.pot","207.pot","208.pot","209.pot","210.pot","211.pot","212.pot","214.pot","215.pot","216.pot","217.pot","218.pot","220.pot","221.pot","222.pot","223.pot","224.pot","225.pot","227.pot","228.pot","230.pot","231.pot","232.pot","233.pot","234.pot","236.pot","237.pot","238.pot","239.pot","243.pot","244.pot","245.pot","247.pot","249.pot","251.pot","252.pot","255.pot","257.pot","258.pot","259.pot","260.pot","261.pot","262.pot","263.pot","264.pot","266.pot","267.pot","269.pot","270.pot","271.pot","272.pot","273.pot","274.pot","275.pot","277.pot","279.pot","280.pot","281.pot","283.pot","284.pot","285.pot","286.pot","287.pot","288.pot","289.pot","290.pot","291.pot","292.pot","294.pot","295.pot","296.pot","298.pot","299.pot","301.pot","303.pot","304.pot","305.pot","306.pot","307.pot","308.pot","309.pot","310.pot","311.pot","312.pot","313.pot","314.pot","315.pot","316.pot","318.pot","319.pot","320.pot","321.pot","322.pot","323.pot","324.pot","325.pot","327.pot","328.pot","329.pot","330.pot","331.pot","332.pot","333.pot","334.pot","335.pot","336.pot","337.pot","338.pot","339.pot","340.pot","342.pot","343.pot","344.pot","346.pot","347.pot","348.pot","349.pot","350.pot","352.pot","353.pot","354.pot","355.pot","356.pot","357.pot","358.pot","359.pot","360.pot","362.pot","363.pot","364.pot","365.pot","366.pot","367.pot","368.pot","369.pot","370.pot","372.pot","373.pot","374.pot","375.pot","378.pot","379.pot","380.pot","381.pot","383.pot","384.pot","385.pot","387.pot","388.pot","389.pot","390.pot","392.pot","393.pot","395.pot","397.pot","399.pot","401.pot","402.pot","403.pot","404.pot","405.pot","406.pot","407.pot","408.pot","409.pot","411.pot","412.pot","413.pot","414.pot","415.pot","416.pot","417.pot","418.pot","420.pot","501.pot","502.pot","503.pot","504.pot","505.pot","506.pot","507.pot","508.pot","509.pot","510.pot","511.pot","512.pot","513.pot","514.pot","515.pot","516.pot","517.pot","518.pot","519.pot","520.pot","521.pot","522.pot","523.pot","524.pot","525.pot","526.pot","527.pot","528.pot","529.pot","530.pot","531.pot","532.pot","533.pot","534.pot","535.pot","536.pot","537.pot","538.pot","539.pot","540.pot","541.pot","542.pot","543.pot","544.pot","545.pot","546.pot","547.pot","548.pot","549.pot","550.pot","551.pot","552.pot","553.pot","554.pot","555.pot","556.pot","557.pot","558.pot","559.pot","560.pot","561.pot","562.pot","563.pot","564.pot","565.pot","566.pot","567.pot","568.pot","569.pot","570.pot","571.pot","572.pot","573.pot","574.pot","575.pot","576.pot","577.pot","578.pot","579.pot","580.pot","581.pot","582.pot","583.pot","584.pot","585.pot","586.pot","587.pot","588.pot","589.pot","590.pot","591.pot","592.pot","593.pot","594.pot","595.pot","596.pot","597.pot","598.pot","599.pot","600.pot","601.pot","602.pot","603.pot","604.pot","605.pot","606.pot","607.pot","608.pot","609.pot","610.pot","611.pot","612.pot","613.pot","614.pot","615.pot","616.pot","617.pot","618.pot","619.pot","620.pot","621.pot","622.pot","623.pot","624.pot","625.pot","626.pot","627.pot","628.pot","629.pot","630.pot","631.pot","632.pot","633.pot","634.pot","635.pot","636.pot","637.pot","638.pot","639.pot","640.pot","641.pot","642.pot","643.pot","644.pot","645.pot","646.pot","647.pot","648.pot","649.pot","650.pot","651.pot","652.pot","653.pot","654.pot","655.pot","656.pot","657.pot","658.pot","659.pot","660.pot","661.pot","662.pot","663.pot","664.pot","665.pot","666.pot","667.pot","668.pot","669.pot","670.pot","671.pot","672.pot","673.pot","674.pot","675.pot","676.pot","677.pot","678.pot","679.pot","680.pot","681.pot","682.pot","683.pot","684.pot","685.pot","686.pot","687.pot","688.pot","689.pot","690.pot","691.pot","692.pot","693.pot","694.pot","695.pot","696.pot","697.pot","698.pot","699.pot","700.pot","701.pot","702.pot","703.pot","704.pot","705.pot","706.pot","707.pot","708.pot","709.pot","710.pot","711.pot","712.pot","713.pot","714.pot","715.pot","716.pot","717.pot","718.pot","719.pot","720.pot","721.pot","722.pot","723.pot","724.pot","725.pot","726.pot","727.pot","728.pot","729.pot","730.pot","731.pot","732.pot","733.pot","734.pot","735.pot","736.pot","737.pot","738.pot","739.pot","740.pot"],
            "PotTest": ["006.pot","013.pot","020.pot","029.pot","038.pot","042.pot","046.pot","058.pot","060.pot","062.pot","069.pot","073.pot","082.pot","084.pot","097.pot","104.pot","108.pot","111.pot","113.pot","121.pot","124.pot","1241.pot","1242.pot","1243.pot","1244.pot","1245.pot","1246.pot","1247.pot","1248.pot","1249.pot","125.pot","1250.pot","1251.pot","1252.pot","1253.pot","1254.pot","1255.pot","1256.pot","1257.pot","1258.pot","1259.pot","1260.pot","1261.pot","1262.pot","1263.pot","1264.pot","1265.pot","1266.pot","1267.pot","1268.pot","1269.pot","1270.pot","1271.pot","1272.pot","1273.pot","1274.pot","1275.pot","1276.pot","1277.pot","1278.pot","1279.pot","1280.pot","1281.pot","1282.pot","1283.pot","1284.pot","1285.pot","1286.pot","1287.pot","1288.pot","1289.pot","1290.pot","1291.pot","1292.pot","1293.pot","1294.pot","1295.pot","1296.pot","1297.pot","1298.pot","1299.pot","1300.pot","132.pot","136.pot","147.pot","150.pot","153.pot","156.pot","157.pot","158.pot","160.pot","168.pot","169.pot","172.pot","173.pot","181.pot","187.pot","189.pot","193.pot","194.pot","195.pot","202.pot","205.pot","213.pot","219.pot","226.pot","229.pot","235.pot","240.pot","241.pot","242.pot","246.pot","248.pot","250.pot","253.pot","254.pot","256.pot","265.pot","268.pot","276.pot","278.pot","282.pot","293.pot","297.pot","300.pot","302.pot","317.pot","326.pot","341.pot","345.pot","351.pot","361.pot","371.pot","376.pot","377.pot","382.pot","386.pot","391.pot","394.pot","396.pot","398.pot","400.pot","410.pot","419.pot","741.pot","742.pot","743.pot","744.pot","745.pot","746.pot","747.pot","748.pot","749.pot","750.pot","751.pot","752.pot","753.pot","754.pot","755.pot","756.pot","757.pot","758.pot","759.pot","760.pot","761.pot","762.pot","763.pot","764.pot","765.pot","766.pot","767.pot","768.pot","769.pot","770.pot","771.pot","772.pot","773.pot","774.pot","775.pot","776.pot","777.pot","778.pot","779.pot","780.pot","781.pot","782.pot","783.pot","784.pot","785.pot","786.pot","787.pot","788.pot","789.pot","790.pot","791.pot","792.pot","793.pot","794.pot","795.pot","796.pot","797.pot","798.pot","799.pot","800.pot"],
            "PotSimple": ["002.pot","007.pot","009.pot","011.pot"],
            "PotSimpleTest": ["003.pot"],
        }

        # TODO 
        
        for key in FILE_LIST:
            files = FILE_LIST[key]
            print(f"开始下载{key}数据集")
            with alive_bar(len(files)) as bar:
                for f in files:
                    full_path = f"{self.__base_dir}/{key}/{f}"
                    full_url = f"{self.__download_url}/{key}/{f}"
                    my_wget.get(full_url, full_path)
                    bar()
            print(f"下载完毕")


def main():
    PotDownloader().start()
    pass

if __name__ == '__main__':
    main()