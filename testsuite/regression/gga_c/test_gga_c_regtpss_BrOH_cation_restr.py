
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_regtpss_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_regtpss", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.101341328186516e-02, -5.101426119304649e-02, -5.101667121474810e-02, -5.100408507737855e-02, -5.101087283404729e-02, -5.101087283404729e-02, -4.165335054637635e-02, -4.165852002754497e-02, -4.177826625114369e-02, -4.157048595773154e-02, -4.165956444042253e-02, -4.165956444042253e-02, -2.835819887232681e-02, -2.812414959516100e-02, -2.249904965889717e-02, -2.275297164637714e-02, -2.290117321186963e-02, -2.290117321186963e-02, -7.444076330814064e-03, -8.072697459306378e-03, -3.082941779605377e-02, -2.616373696916704e-03, -4.373206385296569e-03, -4.373206385296575e-03, -1.549534976271309e-08, -2.061069852491897e-08, -1.708891783601428e-05, -1.151538005218912e-09, -4.176311993090078e-09, -4.176311993090078e-09, -5.882987251585943e-02, -5.904121668343232e-02, -5.883858044355243e-02, -5.902515215911476e-02, -5.893739626224556e-02, -5.893739626224556e-02, -2.101355521378115e-02, -2.147566222462474e-02, -1.997993181244593e-02, -2.037515980952371e-02, -2.178995951357735e-02, -2.178995951357735e-02, -3.975882486493495e-02, -5.641059624458314e-02, -3.718112845663085e-02, -5.190542678862650e-02, -4.162908708060425e-02, -4.162908708060425e-02, -5.927527374104557e-04, -3.910098868370505e-03, -4.735394877378448e-04, -7.210096258488657e-02, -1.471372232278936e-03, -1.471372232278936e-03, -4.559380305946524e-10, -1.220654315627129e-09, -2.206806831041050e-09, -1.383033151232092e-04, -2.440268272719244e-09, -2.440268272719244e-09, -6.065930417734847e-02, -5.546840404720647e-02, -5.720289954960510e-02, -5.871071552221934e-02, -5.794831574286449e-02, -5.794831574286449e-02, -6.173139773217909e-02, -2.827274380593255e-02, -3.562791979289642e-02, -4.460046248110451e-02, -3.989543741118147e-02, -3.989543741118147e-02, -5.633228682648240e-02, -6.636801638020001e-03, -1.112096143787445e-02, -2.477437928507215e-02, -1.716781797170555e-02, -1.716781797170555e-02, -2.759624639428534e-02, -1.302126761007707e-05, -4.567136699361613e-05, -2.969519815383956e-02, -4.186706331831820e-04, -4.186706331831786e-04, -5.811251103080280e-08, -1.189650165760798e-11, -1.956208447804507e-10, -3.257307456932156e-04, -2.150260944880716e-09, -2.150260947049120e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_regtpss_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_regtpss", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.272009685307518e-01, -1.272020060626669e-01, -1.272049642002706e-01, -1.271895624327116e-01, -1.271978677010286e-01, -1.271978677010286e-01, -1.093618893031029e-01, -1.093680144723726e-01, -1.095097969768441e-01, -1.092646356334233e-01, -1.093694003283056e-01, -1.093694003283056e-01, -7.921903544079498e-02, -7.894508919194518e-02, -7.111072362188260e-02, -7.157758240479585e-02, -7.179408712375912e-02, -7.179408712375912e-02, -3.119791610478052e-02, -3.307428009718517e-02, -8.369424095415774e-02, -1.328158578347055e-02, -2.053996809499978e-02, -2.053996809499979e-02, -1.006446916389403e-07, -1.337746235569410e-07, -1.065153623012548e-04, -7.526379691928495e-09, -2.723001519275291e-08, -2.723001519468191e-08, -1.294814410090809e-01, -1.296636144268456e-01, -1.294889774951232e-01, -1.296498109277719e-01, -1.295742253099099e-01, -1.295742253099099e-01, -7.398088521899246e-02, -7.499084440423665e-02, -7.169236903297856e-02, -7.259075810062306e-02, -7.565056473298824e-02, -7.565056473298824e-02, -8.435564319176858e-02, -8.307071628307923e-02, -8.193405056206642e-02, -8.100061027684821e-02, -8.505330012811778e-02, -8.505330012811778e-02, -3.391790498372579e-03, -1.892609508119444e-02, -2.737774772799428e-03, -1.177787654856490e-01, -7.914823936870994e-03, -7.914823936870994e-03, -2.986975632251268e-09, -7.979688600200390e-09, -1.445754601725101e-08, -8.327575016451464e-04, -1.596211495268334e-08, -1.596211495174891e-08, -7.573987408936822e-02, -8.019905643778126e-02, -7.890192738978220e-02, -7.761844286267507e-02, -7.828568845110041e-02, -7.828568845110041e-02, -7.308872924703602e-02, -7.507160361946542e-02, -8.031579679508084e-02, -8.215627849725157e-02, -8.175982010536716e-02, -8.175982010536716e-02, -8.506563174756855e-02, -2.926530060049495e-02, -4.281020180140100e-02, -6.688776626523336e-02, -5.598572514267818e-02, -5.598572514267818e-02, -7.326874488683012e-02, -8.137766267235669e-05, -2.809569258838143e-04, -6.917264753059316e-02, -2.430753553628128e-03, -2.430753553628155e-03, -3.756213401616646e-07, -7.849628493860015e-11, -1.284958511076807e-09, -1.910371781607466e-03, -1.407477186685804e-08, -1.407477186267458e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_regtpss_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_regtpss", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.928056470794981e-10, 1.928091447564828e-10, 1.928139659489046e-10, 1.927621590623646e-10, 1.927908263335486e-10, 1.927908263335486e-10, 1.080116988593695e-06, 1.080338508019554e-06, 1.085347046874887e-06, 1.075659208722899e-06, 1.080254942689301e-06, 1.080254942689301e-06, 1.795562956327370e-03, 1.782646592104725e-03, 1.454607025758747e-03, 1.420957139832721e-03, 1.446917399171454e-03, 1.446917399171454e-03, 1.768148780149254e-01, 1.847465638003828e-01, 9.707599117028184e-04, 1.461492731209552e-01, 1.769424335537020e-01, 1.769424335537021e-01, 1.219892013129538e-02, 1.430362918674935e-02, 5.356349794005606e-02, 4.485263507346354e-03, 9.241671958687771e-03, 9.241671959570985e-03, 2.832052918216029e-07, 2.850741862315658e-07, 2.832798932765815e-07, 2.849295727422399e-07, 2.841559048661225e-07, 2.841559048661225e-07, 6.159509144503637e-06, 6.142622050823575e-06, 5.860986089606750e-06, 5.848280283107834e-06, 6.299705713483579e-06, 6.299705713483579e-06, 5.785695175470621e-03, 7.162576635857395e-03, 7.538359574053721e-03, 1.094179125047295e-02, 5.775797467327897e-03, 5.775797467327897e-03, 9.531618232884959e-02, 6.887414498956615e-02, 9.973619353526450e-02, 5.984102328359404e-05, 1.689589841571022e-01, 1.689589841571022e-01, 4.437916986749038e-03, 5.520883357771357e-03, 5.594831795458050e-02, 1.111102204696921e-01, 2.413971573987092e-02, 2.413971574071670e-02, 1.193106628121097e-02, 1.032167291363346e-02, 1.083491222352602e-02, 1.130098996574306e-02, 1.106304270468161e-02, 1.106304270468161e-02, 1.406401565647270e-02, 7.922487365917030e-03, 9.024601341183840e-03, 1.050624705328337e-02, 9.728349349621255e-03, 9.728349349621255e-03, 5.706859584113859e-03, 4.810103458432971e-02, 4.187007181155688e-02, 3.459085576287740e-02, 3.997106601809781e-02, 3.997106601809785e-02, 1.136466103216749e-02, 4.436983711868766e-02, 6.208654013678876e-02, 5.352556485139843e-02, 1.773671064027639e-01, 1.773671064027658e-01, 1.461539088725043e-02, 6.240284063979563e-03, 7.718859908393574e-03, 1.660043236043836e-01, 3.017405179627611e-02, 3.017405176418282e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05