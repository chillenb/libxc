
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_kcisk_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.177477310850274e-02, -4.177519135081079e-02, -4.177568227002115e-02, -4.176945001777603e-02, -4.177291075250371e-02, -4.177291075250371e-02, -4.370911588519509e-02, -4.371206693237712e-02, -4.377399828504349e-02, -4.359689688640530e-02, -4.367940460849826e-02, -4.367940460849826e-02, -3.471199412951580e-02, -3.453711500554624e-02, -3.028261894954551e-02, -3.018983504027446e-02, -3.031572733459440e-02, -3.031572733459440e-02, -1.317129944961679e-02, -1.367954797739999e-02, -3.864360635696695e-02, -8.682592056974234e-03, -5.697617143222825e-03, -5.697617143222832e-03, -3.970668726395811e-05, -4.513616706792626e-05, -9.952125104599975e-04, -1.304998812850939e-05, -1.340635640973991e-05, -1.340635640973993e-05, -4.255088852018711e-02, -4.264273876847101e-02, -4.255161718949899e-02, -4.263280999085067e-02, -4.259921468527595e-02, -4.259921468527595e-02, -2.520733568956782e-02, -2.528505023458173e-02, -2.446657643298628e-02, -2.452149321982615e-02, -2.562116993378431e-02, -2.562116993378431e-02, -3.661910661436206e-02, -5.120866674654106e-02, -3.736269758378846e-02, -5.029798075576322e-02, -3.738133117551937e-02, -3.738133117551934e-02, -4.519714896075950e-03, -1.057854615404685e-02, -4.092280580836481e-03, -6.917434665369168e-02, -6.389154158888222e-03, -6.389154158888222e-03, -9.348005538219818e-06, -1.384928781657399e-05, -1.963464497848634e-05, -2.351062606722292e-03, -1.626577765754047e-05, -1.626577765754043e-05, -4.566796223709702e-02, -3.825450593165072e-02, -3.990038169059348e-02, -4.184178263570338e-02, -4.078638621219500e-02, -4.078638621219500e-02, -5.540186012271253e-02, -2.682377105517253e-02, -2.985877670857848e-02, -3.394657716120846e-02, -3.180658966039335e-02, -3.180658966039336e-02, -5.224127777252122e-02, -1.370473154520981e-02, -1.780312736409821e-02, -2.809635912342429e-02, -2.220600019895491e-02, -2.220600019895491e-02, -2.904981718720957e-02, -8.730363933813252e-04, -1.466087772823823e-03, -2.990123579146943e-02, -3.767446681135103e-03, -3.767446681135093e-03, -7.444514534025978e-05, -2.367366708723720e-06, -6.461225865361195e-06, -3.360672997663246e-03, -1.636043179444128e-05, -1.636043179444127e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_kcisk_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.005808429757421e-01, -1.005750297983562e-01, -1.005815155016228e-01, -1.005759620594568e-01, -1.005835695550078e-01, -1.005766300367481e-01, -1.005701577147026e-01, -1.005661562932985e-01, -1.005762489311694e-01, -1.005732097602265e-01, -1.005762489311694e-01, -1.005732097602265e-01, -9.594739019585878e-02, -9.596135062918633e-02, -9.595043468862298e-02, -9.596643426771580e-02, -9.604665718172792e-02, -9.604086308387726e-02, -9.579688011401875e-02, -9.580477902152777e-02, -9.598741186914371e-02, -9.583986584395990e-02, -9.598741186914371e-02, -9.583986584395990e-02, -7.150397362532122e-02, -7.111911133222049e-02, -7.134874616280855e-02, -7.089025593048241e-02, -6.566327365561217e-02, -6.618336719503042e-02, -6.592560948336315e-02, -6.567153289350444e-02, -6.364795977124103e-02, -6.872862722485588e-02, -6.364795977124103e-02, -6.872862722485588e-02, -3.586035600987038e-02, -3.264127647883069e-02, -3.697811492965086e-02, -3.326040882937201e-02, -7.982496080491537e-02, -7.568194545994152e-02, -2.549255352367782e-02, -2.437652758248857e-02, -1.416616004987740e-02, -4.472068081665891e-02, -1.416616004987742e-02, -4.472068081665890e-02, -1.598898183089969e-04, -1.463045523702427e-04, -1.828493062854791e-04, -1.652699813402402e-04, -3.741389837893269e-03, -3.438719679155700e-03, -4.941656168625541e-05, -5.056118958761926e-05, -4.467037164307546e-05, -1.062308900183641e-04, -4.467037164307546e-05, -1.062308900183643e-04, -9.740183873317418e-02, -9.743890781946994e-02, -9.752432413052020e-02, -9.756408005967022e-02, -9.740170895098268e-02, -9.744177740225529e-02, -9.751194797253265e-02, -9.755045243192970e-02, -9.746712553201520e-02, -9.750357383484252e-02, -9.746712553201520e-02, -9.750357383484252e-02, -6.302502025371266e-02, -6.296802075480672e-02, -6.328303672272144e-02, -6.326321860993037e-02, -6.164120354289363e-02, -6.139406979786797e-02, -6.187595428871996e-02, -6.161102431793199e-02, -6.365227423327852e-02, -6.410756459565008e-02, -6.365227423327852e-02, -6.410756459565008e-02, -6.779041598611689e-02, -6.796308266457844e-02, -7.486382665250606e-02, -7.479527836559319e-02, -7.104275960849303e-02, -6.525049864467848e-02, -7.590191692375439e-02, -7.036695281780250e-02, -6.444081179045467e-02, -7.284677370901375e-02, -6.444081179045463e-02, -7.284677370901371e-02, -1.394508161286945e-02, -1.373932619281116e-02, -2.987015749700760e-02, -2.963308222423255e-02, -1.323957790207693e-02, -1.207293773340833e-02, -1.090600509283271e-01, -1.091564658277487e-01, -2.033706047826715e-02, -1.786686275337655e-02, -2.033706047826715e-02, -1.786686275337654e-02, -3.616699507577622e-05, -3.454101896770719e-05, -5.300800219627257e-05, -5.229610138477062e-05, -7.836499200942386e-05, -7.190610444504449e-05, -7.793976134714119e-03, -7.640393604843988e-03, -8.879818060607850e-05, -5.320741954813895e-05, -8.879818060607838e-05, -5.320741954813888e-05, -6.970043795516520e-02, -6.915273725508694e-02, -6.669003584798699e-02, -6.613861574110856e-02, -6.757237993931349e-02, -6.702982051480663e-02, -6.844659064373965e-02, -6.788680150385794e-02, -6.799439115029944e-02, -6.744280365231903e-02, -6.799439115029944e-02, -6.744280365231903e-02, -7.103327666400609e-02, -7.050025970388825e-02, -5.645738277512754e-02, -5.588868504039208e-02, -5.949677143635531e-02, -5.890006597644560e-02, -6.277707386116947e-02, -6.228526772404437e-02, -6.115392698491396e-02, -6.057945210041471e-02, -6.115392698491396e-02, -6.057945210041472e-02, -7.696923126239122e-02, -7.667870461114420e-02, -3.650772266516491e-02, -3.611102213914939e-02, -4.391174159442212e-02, -4.301424619339708e-02, -5.661145535671507e-02, -5.609524608752864e-02, -4.955869429915989e-02, -4.958630133807888e-02, -4.955869429915990e-02, -4.958630133807888e-02, -5.795605980062183e-02, -5.678363498116119e-02, -3.193832395789097e-03, -3.166026707434430e-03, -4.902095898482658e-03, -4.632769145747151e-03, -5.778606468828654e-02, -5.461998367362637e-02, -1.233680900040527e-02, -1.107039382192266e-02, -1.233680900040525e-02, -1.107039382192264e-02, -2.879607201166576e-04, -2.746626628966769e-04, -8.837334347347392e-06, -8.815040859571390e-06, -2.609566314737760e-05, -2.380989718379851e-05, -1.077791210130494e-02, -1.036769712732919e-02, -8.307519631441221e-05, -5.455726774471687e-05, -8.307519631441217e-05, -5.455726774471685e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_kcisk_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.926929386731344e-10, 3.249000840039912e-10, 2.927258563988885e-10, 2.927017462420922e-10, 3.249072267112847e-10, 2.927332251099646e-10, 2.927186397323691e-10, 3.249187920325246e-10, 2.927565122384058e-10, 2.926070553746181e-10, 3.248129601806282e-10, 2.926286624802181e-10, 2.926820636632333e-10, 3.248712764492142e-10, 2.926835910530251e-10, 2.926820636632333e-10, 3.248712764492142e-10, 2.926835910530251e-10, 1.185835317587609e-06, 1.750211353399554e-06, 1.186075407870875e-06, 1.186150731129452e-06, 1.750580057857059e-06, 1.186378618678690e-06, 1.193268323230323e-06, 1.758926068444677e-06, 1.193874967301778e-06, 1.182511012214126e-06, 1.742851245046208e-06, 1.182819865483458e-06, 1.185881883749943e-06, 1.750449030315692e-06, 1.189223751192459e-06, 1.185881883749943e-06, 1.750449030315692e-06, 1.189223751192459e-06, 1.755849488471091e-03, 2.710585774918458e-03, 1.778519896539361e-03, 1.740973979403514e-03, 2.693656347118524e-03, 1.767381487000210e-03, 1.454910903790165e-03, 2.273975117889498e-03, 1.436261316588002e-03, 1.423794594491046e-03, 2.216335424706540e-03, 1.434817638648441e-03, 1.567624649502061e-03, 2.245160129576953e-03, 1.344789954797078e-03, 1.567624649502061e-03, 2.245160129576953e-03, 1.344789954797078e-03, 2.318646286055061e-01, 3.507582033158321e-01, 2.385181127626737e-01, 2.361806373213059e-01, 3.522820028500470e-01, 2.437785044016270e-01, 8.363609766717259e-04, 1.456249520648073e-03, 8.620079388652144e-04, 2.763484993572184e-01, 4.873960841408359e-01, 2.794103243331620e-01, 2.166620049927170e-01, 3.385156371586861e-01, 1.986660885443623e-01, 2.166620049927169e-01, 3.385156371586861e-01, 1.986660885443623e-01, 1.868624020203208e+01, 3.729886305740945e+01, 1.868624403322199e+01, 1.869543278575808e+01, 3.731140162006727e+01, 1.869586092558994e+01, 1.753910478352442e+00, 3.463972139578223e+00, 1.755041436037746e+00, 2.982771435395014e+01, 5.960111302313992e+01, 2.982706672693933e+01, 2.070188596856347e+01, 4.133319261081638e+01, 2.077630728184674e+01, 2.070188596856352e+01, 4.133319261081648e+01, 2.077630728184677e+01, 5.115841318050139e-07, 4.789676177494759e-07, 5.118661177008237e-07, 5.167702814694074e-07, 4.822023521218240e-07, 5.169350974559142e-07, 5.118685981012981e-07, 4.790968875124962e-07, 5.120652363255821e-07, 5.163566384567902e-07, 4.819522117530358e-07, 5.166268070153316e-07, 5.142091635278343e-07, 4.806129963633914e-07, 5.144118974221399e-07, 5.142091635278343e-07, 4.806129963633914e-07, 5.144118974221399e-07, 7.339610819725815e-06, 9.969809869345877e-06, 7.350157261765679e-06, 7.368607699307388e-06, 9.911421286590110e-06, 7.377992884970347e-06, 7.005164414313446e-06, 9.567280627564464e-06, 7.053333611245784e-06, 7.029359079816548e-06, 9.516124428718085e-06, 7.077912939963857e-06, 7.562987634138621e-06, 1.014167284179772e-05, 7.481755607169795e-06, 7.562987634138621e-06, 1.014167284179772e-05, 7.481755607169795e-06, 8.874600470323639e-03, 8.259458366033385e-03, 9.047212771077263e-03, 1.672430526649924e-02, 9.846204232932908e-03, 1.731803797777690e-02, 9.748233638514929e-03, 1.064067318375358e-02, 1.008985663735796e-02, 1.788646286380096e-02, 1.439941844442556e-02, 1.754887804444310e-02, 9.238644064131388e-03, 8.205638738074717e-03, 9.696838917857453e-03, 9.238644064131388e-03, 8.205638738074714e-03, 9.696838917857451e-03, 3.820020759105478e-01, 7.132754636756030e-01, 3.825143654850459e-01, 1.174750788113465e-01, 1.969513674371884e-01, 1.173391973532292e-01, 4.482177063798475e-01, 8.424708405720440e-01, 4.490936493918738e-01, 8.140509461184264e-05, 9.893294881456513e-05, 8.152583330189171e-05, 4.087300731490963e-01, 7.460302067285165e-01, 4.276280088766499e-01, 4.087300731490963e-01, 7.460302067285165e-01, 4.276280088766500e-01, 5.091725186820180e+01, 1.017713684870861e+02, 5.091199962292200e+01, 3.588574373114296e+01, 7.170783291047076e+01, 3.588155386956774e+01, 2.903547599364937e+02, 5.797687543718996e+02, 2.904350979873125e+02, 9.707789699666718e-01, 1.867626318853139e+00, 9.719614381889057e-01, 9.901256786206223e+01, 1.978211657316054e+02, 9.909471939984560e+01, 9.901256786206200e+01, 1.978211657316049e+02, 9.909471939984533e+01, 1.231722264459956e-01, 1.519111576761105e-02, 1.248574395451128e-01, 5.467655931810393e-02, 1.359173759935654e-02, 5.507509145161078e-02, 6.896384954766190e-02, 1.411790979278940e-02, 6.967041401161138e-02, 8.689183406126651e-02, 1.458262033677941e-02, 8.759711943507804e-02, 7.701679195607530e-02, 1.434693318356384e-02, 7.772524467928908e-02, 7.701679195607530e-02, 1.434693318356384e-02, 7.772524467928905e-02, 1.330162785092925e-01, 1.735152833330192e-02, 1.348506916101247e-01, 1.162919633085428e-02, 1.161765409568197e-02, 1.164522685004113e-02, 1.592639043536303e-02, 1.264346676532583e-02, 1.601706279019903e-02, 2.625332353055579e-02, 1.413452218650358e-02, 2.617019552857844e-02, 1.985190241008848e-02, 1.335517036332495e-02, 1.982464159358287e-02, 1.985190241008848e-02, 1.335517036332495e-02, 1.982464159358287e-02, 1.132223367522918e-02, 8.025724969885991e-03, 1.187280091927659e-02, 6.946276581140733e-02, 1.085290083689479e-01, 6.948307247781466e-02, 5.294164158615846e-02, 7.530953343595305e-02, 5.334745276329933e-02, 4.351782065456283e-02, 4.610849918693850e-02, 4.323061297468821e-02, 4.888991145094108e-02, 6.044648168387644e-02, 4.922855975144517e-02, 4.888991145094107e-02, 6.044648168387644e-02, 4.922855975144518e-02, 1.515716554037385e-02, 1.623993701718884e-02, 1.536559078748826e-02, 1.700280235784764e+00, 3.359664252825067e+00, 1.700145797857624e+00, 1.053996406176921e+00, 2.065336204577509e+00, 1.055418280854814e+00, 7.365341034578783e-02, 6.571467133203973e-02, 7.990460494149727e-02, 8.250107930565325e-01, 1.558382927654625e+00, 8.404014221033682e-01, 8.250107930565312e-01, 1.558382927654622e+00, 8.404014221033667e-01, 1.063654390447935e+01, 2.123039567788604e+01, 1.063694882598377e+01, 6.623988174926734e+02, 1.324578091104081e+03, 6.624829102219356e+02, 1.514511982780304e+02, 3.026655580881308e+02, 1.514631549460069e+02, 9.006787824094596e-01, 1.693684819541966e+00, 9.030870486992095e-01, 1.386576576200713e+02, 2.766448223994853e+02, 1.385153021993991e+02, 1.386576576200717e+02, 2.766448223994861e+02, 1.385153021993995e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_kcisk_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_kcisk_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.845812445625717e-05, -1.846718637857590e-05, -1.845900959382616e-05, -1.846782820286249e-05, -1.846175452330337e-05, -1.847178758909395e-05, -1.845184978888912e-05, -1.845912995674501e-05, -1.845869103697044e-05, -1.846411710619422e-05, -1.845869103697044e-05, -1.846411710619422e-05, -9.490032312039767e-05, -9.479452400430523e-05, -9.495253019894427e-05, -9.482225067004042e-05, -9.589963625626702e-05, -9.604968982439739e-05, -9.583942684498364e-05, -9.577256732065964e-05, -9.482155340306879e-05, -9.663540830158521e-05, -9.482155340306879e-05, -9.663540830158521e-05, -7.150229868774087e-04, -7.324316337411361e-04, -7.114292585836429e-04, -7.294039063456037e-04, -6.455759733195581e-04, -6.471281978712801e-04, -7.156677789289413e-04, -7.351377621079282e-04, -7.186458393924199e-04, -6.589822582034490e-04, -7.186458393924199e-04, -6.589822582034490e-04, -6.626341750094913e-03, -8.381550208722388e-03, -6.931445909359270e-03, -8.985391378941049e-03, -1.500605645393116e-04, -1.879995764520346e-04, -2.077938211781242e-03, -2.645181337996490e-03, -6.174745745801617e-03, -8.426680693416198e-04, -6.174745745801610e-03, -8.426680693416198e-04, -1.042986974685711e-05, -1.182206728848213e-05, -1.187229199100503e-05, -1.379695158550692e-05, -1.717033939107929e-04, -2.008052197857752e-04, -2.762177037534625e-06, -2.698455878329098e-06, -7.886630513201705e-06, -4.371020603957391e-06, -7.886630513201705e-06, -4.371020603957392e-06, -5.606532709867141e-04, -5.610795036550049e-04, -5.672043180278496e-04, -5.674154983409870e-04, -5.611544823471630e-04, -5.613967208748811e-04, -5.667702295445363e-04, -5.671471357628906e-04, -5.638889515539068e-04, -5.642315484923593e-04, -5.638889515539068e-04, -5.642315484923593e-04, -2.967283411872251e-04, -2.993091218894105e-04, -3.149665753221255e-04, -3.170882324928633e-04, -2.855799207884546e-04, -2.911120068856387e-04, -3.011842513017648e-04, -3.072099688527982e-04, -3.158670376468358e-04, -3.100196507538572e-04, -3.158670376468358e-04, -3.100196507538572e-04, -7.801751747523959e-03, -8.147928109582127e-03, -9.416823337777417e-03, -9.971148057164309e-03, -4.326743194326976e-03, -5.629260558159108e-03, -3.874827420379790e-03, -4.981728880331975e-03, -9.616675736807516e-03, -8.153077302275762e-03, -9.616675736807518e-03, -8.153077302275760e-03, -9.769947999237636e-04, -9.892783591394598e-04, -3.179652403243929e-03, -3.175699621816866e-03, -8.056857026867899e-04, -9.496051081462950e-04, -4.612401282760084e-04, -4.618143059157588e-04, -1.706234614092169e-03, -2.696551500396208e-03, -1.706234614092170e-03, -2.696551500396211e-03, -1.130517325298140e-06, -1.123827207785478e-06, -2.425062792912158e-06, -2.202900134721047e-06, -9.746508567129855e-06, -1.199355566908912e-05, -4.910860034801819e-04, -5.630235439952724e-04, -1.917402292378995e-06, -1.147939568169565e-05, -1.917402292378995e-06, -1.147939568169563e-05, -1.126200161523754e-01, -1.153032883911060e-01, -7.130547202539714e-02, -7.267994446807732e-02, -8.401971804145537e-02, -8.587009000058761e-02, -9.684520054295065e-02, -9.884819666694830e-02, -9.017994379482268e-02, -9.210996181256426e-02, -9.017994379482268e-02, -9.210996181256423e-02, -4.426151248273188e-02, -4.630217598910785e-02, -1.055729782661232e-02, -1.067746677287369e-02, -1.645615277916390e-02, -1.674217206328580e-02, -2.868265986013754e-02, -2.880780604080840e-02, -2.114939278311404e-02, -2.138723356946749e-02, -2.114939278311403e-02, -2.138723356946748e-02, -6.162212186035974e-03, -6.766546001671947e-03, -4.331605737573590e-03, -4.374034460556666e-03, -5.632358920291751e-03, -5.847407694308658e-03, -8.627953205570667e-03, -8.541995294125443e-03, -7.806845747042287e-03, -7.893477455925132e-03, -7.806845747042284e-03, -7.893477455925132e-03, -7.881325727636609e-03, -8.380954095001050e-03, -1.744425181305293e-04, -1.732553023203760e-04, -2.673429500792625e-04, -2.963641079329024e-04, -1.186029283933324e-02, -1.625313783823550e-02, -9.292542251468766e-04, -1.284747168152515e-03, -9.292542251468764e-04, -1.284747168152516e-03, -1.028503153651988e-05, -1.088124254691224e-05, -2.020187433773603e-07, -2.019539938367911e-07, -2.317161104169092e-06, -2.776348196728033e-06, -9.872808797743791e-04, -1.143785884802165e-03, -4.434767772085329e-06, -9.999963935625730e-06, -4.434767772085327e-06, -9.999963935625729e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05