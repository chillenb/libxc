
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_b94_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.177024122159752e-02, -7.177040834271915e-02, -7.177016975482362e-02, -7.176767448254845e-02, -7.176912170716176e-02, -7.176912170716176e-02, -5.015517643876113e-02, -5.015553679904473e-02, -5.015299203059840e-02, -5.003003165235988e-02, -5.009312187706860e-02, -5.009312187706860e-02, -3.362511815230181e-02, -3.346872497075130e-02, -2.951688854635294e-02, -2.915367023705695e-02, -2.950577668367845e-02, -2.950577668367845e-02, -7.308456096854188e-03, -7.693645956135490e-03, -4.864673333424312e-02, -4.886245033360495e-03, -6.110393153064132e-03, -6.110393153063805e-03, -4.241176134437597e-06, -5.164048921020926e-06, -3.282158291725485e-04, -2.101748842446364e-06, -2.971579662568672e-06, -2.971579662568698e-06, -1.029753584484124e-02, -1.923773582009635e-02, -1.136848705386092e-01, -6.047820289424473e-03, -8.953440791778369e-02, -3.104852759888064e-02, -1.095155957237719e-01, -1.854290892504225e-02, -2.537047398856129e-02, -2.548616477505938e-02, -2.620400252408092e-02, -3.451168242417349e-02, -4.189098320548686e-03, -1.018562711534977e-02, -4.349587589093619e-03, -3.040585208290089e-02, -2.523319942863346e-02, -2.130599890083809e-02, -7.553924913503357e-02, -1.808297767972894e-03, -8.250884604683445e-02, -3.994029273800526e-02, -3.919803615167766e-03, -5.507572473449388e-03, -1.980303300438839e-07, -5.981298707484432e-11, -6.232986828527942e-01, -1.001445355494674e-01, -9.959344080068088e-07, -1.374033120316297e+00, -3.613948292122474e-07, -6.800070050683643e-02, -6.807301452245809e-02, -9.580419535585227e-05, -6.810115993537946e-02, -1.401873300642917e-08, -6.870146040901719e-02, -2.120846272469742e-02, -5.238443794646951e-03, -5.539411267956578e-02, -1.880163256317325e-02, -2.737577473802091e-03, -4.234309654414977e-03, -4.403702580783232e-02, -5.279557976075069e-02, -6.614032704564232e-02, -8.821629226999450e-03, -1.135589236986643e-02, -2.045286204347630e-02, -6.589424943974169e-02, -5.166880509626164e-04, -1.997632391324791e-02, -6.370810261663490e-02, -2.253674709012329e-03, -1.167101324045603e+00, -2.308638778619792e+01, -6.359639267683672e-01, -4.366140978596429e-02, -9.434912041876075e-01, -9.434926663655955e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_b94_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.722268369728697e-02, -4.722222212034357e-02, -4.722137360842794e-02, -4.722824698349554e-02, -4.722447341977819e-02, -4.722447341977819e-02, -5.341350673289784e-02, -5.340893937278260e-02, -5.330743338038317e-02, -5.353171265672574e-02, -5.343149585666199e-02, -5.343149585666199e-02, -4.730040522946869e-02, -4.738277911147667e-02, -4.838441676171494e-02, -4.827218094736370e-02, -4.838044483505517e-02, -4.838044483505517e-02, -2.063856254653055e-02, -2.125194047742497e-02, -5.141083901966777e-02, -1.541512936524385e-02, -1.836951989319074e-02, -1.836951989319066e-02, -4.565041324846377e-05, -5.257962090660889e-05, -1.776778212858546e-03, -1.673951708821013e-05, -2.592401859079213e-05, -2.592401859079221e-05, -2.199629601558352e-02, -3.573581912328527e-02, -1.326308941009199e-01, -1.393879445260962e-02, -7.244668055945680e-02, -4.776508182980167e-02, -1.729004074089978e-01, -3.932316256368097e-02, -5.271728848028903e-02, -5.280611846430273e-02, -5.281826861972087e-02, -6.655168632938806e-02, -9.729392179989048e-03, -2.009149457853705e-02, -1.001837790846264e-02, -3.144015930473525e-02, -3.691591199386337e-02, -3.434014309669836e-02, -2.894868487081990e-01, -5.022960410748232e-03, -3.117430123198526e-01, -4.048625800649684e-02, -1.477414736810248e-02, -2.327600400262335e-02, -7.611205266502056e-07, -1.909525851088042e-10, -4.209253131609999e+00, -4.390856367859321e-01, -4.887545091864698e-06, -4.890071507780427e+00, -1.118244716242375e-06, -8.091616661469707e-02, -7.986127585914451e-02, -2.722400385590326e-04, -7.941892070498904e-02, -4.436407562235589e-08, -7.545130097306661e-02, -3.914318880696740e-02, -1.191333932344890e-02, -5.521479055626194e-02, -3.220442374051779e-02, -6.621060010480149e-03, -9.726488634808393e-03, -1.491218372742656e-01, -1.221530539522591e-01, -8.100469150119542e-02, -1.993688926020484e-02, -2.518522255946023e-02, -3.708334015801788e-02, -9.941609140095448e-01, -2.448683167968107e-03, -3.276165603690128e-02, -3.035289088296632e-01, -9.961790573759476e-03, -3.345007013336307e+00, -6.752520923711440e+00, -9.637605718596184e+00, -3.139389978556739e-01, -4.840442658782217e+00, -4.840452546092179e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b94_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.351503147166813e-10, 1.351483722172685e-10, 1.351386599101112e-10, 1.351675492324781e-10, 1.351525646351237e-10, 1.351525646351237e-10, 8.646249814951402e-07, 8.646639636369632e-07, 8.652786956344907e-07, 8.611764960321717e-07, 8.634532921483954e-07, 8.634532921483954e-07, 1.537607227167303e-03, 1.529194025642275e-03, 1.316482446276083e-03, 1.274505769137286e-03, 1.304990027317392e-03, 1.304990027317392e-03, 1.436483355776735e-01, 1.453848607853557e-01, 1.047560757269062e-03, 2.236566180667178e-01, 2.072124467081212e-01, 2.072124467081154e-01, 1.349613388572524e+01, 1.348200125020491e+01, 1.603978244124201e+00, 2.204594484792131e+01, 2.013055670731977e+01, 2.013055670731983e+01, 5.188671375280627e-09, 1.718864682634756e-08, 1.892028619184956e-07, 1.957079117970256e-09, 1.480775783971302e-07, 4.706563177609310e-08, 2.290632825373906e-05, 2.174367122678505e-06, 4.436403372196614e-06, 4.366867237046833e-06, 4.459406842734032e-06, 8.260909996104871e-06, 6.787615054985021e-05, 2.331081127981906e-04, 1.065533319353455e-04, 4.187464831373209e-03, 2.248408122133314e-03, 1.525373162407779e-03, 2.593760895641429e+01, 9.289610107631071e-03, 3.651855823229084e+01, 1.186929745631366e-05, 4.940894508247969e-01, 9.391392357595443e-01, 1.079688055622284e+00, 5.711106668994969e-07, 5.605877092926489e+07, 1.936422780441220e+02, 1.121512839234950e+01, 2.651091572230437e+07, 6.863118461691049e-12, 6.965859660228460e-03, 6.900957012525918e-03, 8.748096133271297e-08, 6.874046703677516e-03, 3.012202770185543e-14, 7.701765493563068e-03, 4.620581989427458e-03, 2.072751022297936e-04, 8.131663801589337e-03, 2.397754511010170e-03, 5.573061823716713e-05, 3.496598854155771e-05, 5.911135687578882e-01, 2.627911836028528e-01, 7.294939388165141e-02, 9.396756926134550e-03, 1.565909863947409e-02, 6.213648499438266e-03, 1.742548650958282e+03, 8.770365865510825e-01, 3.026043683232216e-02, 7.028840718775849e+01, 1.256130856426150e+00, 4.723140052616214e+05, 3.806405293313819e+09, 1.944070033942087e+08, 8.623543230001816e+01, 3.622564589912252e+07, 3.622570166906857e+07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b94_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [2.558342696574744e-06, 2.558313839565118e-06, 2.558201056322948e-06, 2.558629149385164e-06, 2.558402660532678e-06, 2.558402660532678e-06, 9.870164199837608e-05, 9.870531281160093e-05, 9.875339642046329e-05, 9.825470059780834e-05, 9.851425925883376e-05, 9.851425925883376e-05, 1.221334320743616e-03, 1.208688861869308e-03, 9.169411210415061e-04, 9.010934104063618e-04, 9.236337687893268e-04, 9.236337687893268e-04, 1.077239492808743e-03, 1.171082118571220e-03, 1.502067625509696e-03, 7.371621462793840e-04, 9.831985061168787e-04, 9.831985061169498e-04, 8.389638994009273e-07, 1.081636958300318e-06, 5.800805796259240e-05, 5.784157824415334e-07, 7.911305061068110e-07, 7.911305061068178e-07, 1.068328566436600e-06, 4.373687875346699e-06, 3.057417378051139e-08, 3.453995113506660e-07, 1.231171080252249e-05, 1.458056636853481e-05, 1.767682170158155e-08, 3.017948840333295e-05, 6.766074004784295e-05, 6.787796695032776e-05, 7.153925303762934e-05, 1.497034093922937e-04, 1.534665564583770e-05, 8.982395432211314e-05, 1.959960562130653e-05, 1.865538464829104e-03, 1.066126211939650e-03, 6.644867223289553e-04, 6.371674577419094e-05, 4.286661557234306e-05, 1.130533342295539e-06, 2.143208774277714e-04, 8.034087600047302e-04, 1.791386744803513e-03, 9.879044013883046e-09, 2.543075667949580e-15, 8.009027047409239e-13, 3.318802138459269e-05, 1.786618734969635e-07, 2.441454355571549e-09, 3.580328771624822e-13, 1.319148827485747e-12, 4.408129694832572e-15, 9.205063810544590e-09, 1.297177064814865e-11, 1.135818769940124e-15, 1.506756822916249e-14, 1.139860591619426e-03, 3.334121593727618e-05, 4.320842088174614e-04, 6.750532200624326e-04, 8.012256585154310e-06, 1.177961247533155e-05, 2.656299519626829e-07, 1.433367263578723e-07, 2.156838346536028e-05, 3.836532595626052e-04, 7.154279777750631e-04, 1.206944083970577e-03, 2.055949104280382e-05, 7.597259029455557e-05, 2.488001136519659e-03, 3.116243031222052e-06, 5.983785989489873e-04, 1.014452739461178e-07, 3.014429624418592e-06, 2.359265719574636e-09, 2.448214086217639e-03, 5.335768713087769e-08, 1.500099767131772e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b94_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.090311558099363e-05, -2.090299002924322e-05, -2.090231928780614e-05, -2.090418427457208e-05, -2.090322313303903e-05, -2.090322313303903e-05, -5.771039181412603e-04, -5.771336742635563e-04, -5.776728902228894e-04, -5.750386417937188e-04, -5.763680145246318e-04, -5.763680145246318e-04, -7.364392638198563e-03, -7.299899099568468e-03, -5.796920970938309e-03, -5.752997177976954e-03, -5.859716565688146e-03, -5.859716565688146e-03, -1.184451462974498e-02, -1.254064974756184e-02, -7.954680353591001e-03, -8.069258516077165e-03, -1.058556767938403e-02, -1.058556767938373e-02, -4.694384563979125e-05, -5.468160195926188e-05, -1.092493148796711e-03, -1.477149624069610e-05, -2.676198974708993e-05, -2.676198974709000e-05, -1.162339919810997e-05, -3.852634832984056e-05, -4.238543088124815e-04, -4.386401832052209e-06, -3.318081481113913e-04, -1.054633678397644e-04, -2.725091117765738e-03, -2.643472359073355e-04, -5.194443381563333e-04, -5.212512914982691e-04, -5.410461805281626e-04, -1.002270920483948e-03, -2.175913055876592e-04, -9.580620808053438e-04, -2.706088955611902e-04, -1.194574663156972e-02, -7.562011477548822e-03, -5.130247150472017e-03, -3.131800569357174e-01, -7.694692241745212e-04, -3.491243214028744e-01, -1.332585397539196e-03, -9.641755174128542e-03, -1.832654101498431e-02, -3.322876806097938e-07, -3.575067591034422e-13, -1.568889451094555e+01, -5.632511268589205e-01, -5.487555068988935e-06, -1.297177391698077e+01, -2.232320990296548e-11, -2.226971666608122e-02, -2.219686551073688e-02, -2.827927516895413e-07, -2.216564003523462e-02, -9.712968967951543e-14, -2.306859914568209e-02, -8.754658064129131e-03, -4.509500864473099e-04, -2.016956721617220e-02, -5.566143357039792e-03, -1.293729650221114e-04, -1.655861469200656e-04, -9.874762680823064e-02, -7.603947496558137e-02, -4.864311724378483e-02, -4.120945302637527e-03, -6.867293630040044e-03, -9.186699256837546e-03, -1.040671841680359e+00, -1.327267017147875e-03, -1.807699853365744e-02, -4.220813295684984e-01, -7.543055863765597e-03, -4.618114100050240e+00, -4.551002943949209e+01, -2.149655035540701e+01, -4.287995193692701e-01, -1.404288992525509e+01, -1.404291154450280e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05