
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_apbe_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.113600101744060e+01, -2.113602418397447e+01, -2.113620276661457e+01, -2.113585654089439e+01, -2.113602720911755e+01, -2.113602720911755e+01, -3.513987179536617e+00, -3.513942686569282e+00, -3.513000278328130e+00, -3.515306230238290e+00, -3.514018804233252e+00, -3.514018804233252e+00, -7.089377211178836e-01, -7.090429900485533e-01, -7.134169621619102e-01, -7.181605300076073e-01, -7.177812820794269e-01, -7.177812820794269e-01, -2.211326067332763e-01, -2.222650147515352e-01, -8.178623821247246e-01, -1.862736030134526e-01, -2.161005633770642e-01, -2.161005633770642e-01, -1.010690125455097e-02, -1.064265646307716e-02, -5.790491818204822e-02, -5.829257039379523e-03, -8.135150395748207e-03, -8.135150395748207e-03, -5.070115314208404e+00, -5.069254347761213e+00, -5.070087661141834e+00, -5.069327399816089e+00, -5.069672274033886e+00, -5.069672274033886e+00, -2.156625686701071e+00, -2.166188593356184e+00, -2.159063472667937e+00, -2.167524406868243e+00, -2.160720774785818e+00, -2.160720774785818e+00, -5.867811234113309e-01, -6.048489037952596e-01, -5.481978566153249e-01, -5.400256842770652e-01, -5.927480704410735e-01, -5.927480704410735e-01, -1.409585515141401e-01, -2.368881129582729e-01, -1.316830838852396e-01, -1.819387020499923e+00, -1.581853370135963e-01, -1.581853370135963e-01, -4.500584804627185e-03, -5.698419347539768e-03, -4.363574176212518e-03, -9.149145203958776e-02, -5.481225069325212e-03, -5.481225069325213e-03, -5.514285482591409e-01, -5.556922989392126e-01, -5.542164402674673e-01, -5.529711048806453e-01, -5.535952814337337e-01, -5.535952814337337e-01, -5.342941268266870e-01, -5.188544628615586e-01, -5.233939729670520e-01, -5.272230601501001e-01, -5.250705585901032e-01, -5.250705585901032e-01, -6.354374597735408e-01, -2.829875763204230e-01, -3.184912559624573e-01, -3.723591120650533e-01, -3.434346238464013e-01, -3.434346238464013e-01, -4.789862133245958e-01, -5.544053879956740e-02, -7.484376728988312e-02, -3.471853182397788e-01, -1.133068161322608e-01, -1.133068161322608e-01, -1.424101153704198e-02, -1.523274476790352e-03, -3.203277339835466e-03, -1.071639376075472e-01, -5.033689702215422e-03, -5.033689702215418e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_apbe_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.496681574810714e+01, -2.496678655400623e+01, -2.496693346319833e+01, -2.496687204840523e+01, -2.496727642296656e+01, -2.496737290272652e+01, -2.496594790992517e+01, -2.496567932805736e+01, -2.496688765427709e+01, -2.496633154985822e+01, -2.496688765427709e+01, -2.496633154985822e+01, -4.005053575896461e+00, -4.005245625793527e+00, -4.005096980646189e+00, -4.005297007637639e+00, -4.006231473021702e+00, -4.006552721304615e+00, -4.004913014017023e+00, -4.005240148907459e+00, -4.004286681472850e+00, -4.006318852205875e+00, -4.004286681472850e+00, -4.006318852205875e+00, -7.522319336875493e-01, -7.568160084337319e-01, -7.505496902157229e-01, -7.561274322819389e-01, -7.299824483966003e-01, -7.239613619326312e-01, -7.321693116422889e-01, -7.341157839806789e-01, -7.653010276421606e-01, -7.088965968284795e-01, -7.653010276421606e-01, -7.088965968284795e-01, -2.069616208569114e-01, -2.109777741481546e-01, -2.071560886662860e-01, -2.121289281841040e-01, -8.650665936121175e-01, -9.026845187705738e-01, -1.883501699185911e-01, -1.900562292069547e-01, -2.149752308264810e-01, -1.606569079437657e-01, -2.149752308264809e-01, -1.606569079437656e-01, -1.301855773120503e-02, -1.383125719233111e-02, -1.363935488493879e-02, -1.460770365854128e-02, -7.312381857693454e-02, -7.687089624960969e-02, -7.833311263244097e-03, -7.702965557020465e-03, -1.161727917165360e-02, -6.614687884380101e-03, -1.161727917165360e-02, -6.614687884380101e-03, -6.164540261366755e+00, -6.162989147718473e+00, -6.167527663614671e+00, -6.165875209061294e+00, -6.164704196055388e+00, -6.163086878468259e+00, -6.167267554696266e+00, -6.165708728760478e+00, -6.166071743038345e+00, -6.164438738885031e+00, -6.166071743038345e+00, -6.164438738885031e+00, -2.179392385116796e+00, -2.179280766811364e+00, -2.196065312959758e+00, -2.195451400242078e+00, -2.164944594103003e+00, -2.168877599091933e+00, -2.179038926692559e+00, -2.183244673871657e+00, -2.200253921359116e+00, -2.189308710282907e+00, -2.200253921359116e+00, -2.189308710282907e+00, -6.813690796939484e-01, -6.796542625632168e-01, -7.715084336868476e-01, -7.723113437984618e-01, -6.129427866011924e-01, -6.382131841521431e-01, -6.675820494431056e-01, -6.892618687825919e-01, -7.130299102879840e-01, -6.761245094866094e-01, -7.130299102879840e-01, -6.761245094866093e-01, -1.615251108424961e-01, -1.620499068482439e-01, -2.318417427933947e-01, -2.324612826979435e-01, -1.505459908068728e-01, -1.557115946081939e-01, -2.327900425916636e+00, -2.326925678762299e+00, -1.701043447831069e-01, -1.684067934503030e-01, -1.701043447831069e-01, -1.684067934503030e-01, -5.877458799823362e-03, -6.108093808082201e-03, -7.536957383867733e-03, -7.651068222046527e-03, -5.634963806602404e-03, -5.966875435631971e-03, -1.128003545012636e-01, -1.135758153400430e-01, -5.756304396340691e-03, -7.901776900874997e-03, -5.756304396340692e-03, -7.901776900874999e-03, -7.224434813240340e-01, -7.253984466420705e-01, -7.090688292885109e-01, -7.121064588678141e-01, -7.136524414969323e-01, -7.166924893176104e-01, -7.175686109105269e-01, -7.205269671884710e-01, -7.156007841105108e-01, -7.185989061344104e-01, -7.156007841105108e-01, -7.185989061344104e-01, -7.059515939011081e-01, -7.083082849627950e-01, -5.531934211941856e-01, -5.558932815819817e-01, -5.905412561449621e-01, -5.936830733997366e-01, -6.337727632811544e-01, -6.362222184773625e-01, -6.113930505205665e-01, -6.138884045438798e-01, -6.113930505205665e-01, -6.138884045438798e-01, -8.066589805529002e-01, -8.086962264620251e-01, -2.679611851263951e-01, -2.687929079687305e-01, -3.005681312740806e-01, -3.024766727948535e-01, -3.881264345221307e-01, -3.904148091865436e-01, -3.370725442663455e-01, -3.368946004106335e-01, -3.370725442663455e-01, -3.368946004106334e-01, -5.079745063320945e-01, -5.119848982449217e-01, -7.197332696236293e-02, -7.242921820388200e-02, -9.413218742591817e-02, -9.679300795079625e-02, -3.771795794115794e-01, -3.836128066494650e-01, -1.316926291419551e-01, -1.341220085136813e-01, -1.316926291419551e-01, -1.341220085136813e-01, -1.859992468585663e-02, -1.926988815492938e-02, -2.028619118001785e-03, -2.033207396648700e-03, -4.127115438042525e-03, -4.389352305500994e-03, -1.268491973446395e-01, -1.284390976139171e-01, -5.451648296220375e-03, -7.243546748348624e-03, -5.451648296220369e-03, -7.243546748348618e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_apbe_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.627733200449383e-08, 0.000000000000000e+00, -1.627742321003337e-08, -1.627721395721154e-08, 0.000000000000000e+00, -1.627733739798295e-08, -1.627667135996324e-08, 0.000000000000000e+00, -1.627660607654735e-08, -1.627800870075488e-08, 0.000000000000000e+00, -1.627830930850160e-08, -1.627726992812433e-08, 0.000000000000000e+00, -1.627750712142641e-08, -1.627726992812433e-08, 0.000000000000000e+00, -1.627750712142641e-08, -2.176718855281491e-05, 0.000000000000000e+00, -2.177463305585957e-05, -2.176769978135748e-05, 0.000000000000000e+00, -2.177571102052296e-05, -2.178581609115592e-05, 0.000000000000000e+00, -2.179005717780964e-05, -2.174036330358731e-05, 0.000000000000000e+00, -2.174602220844583e-05, -2.177928597310126e-05, 0.000000000000000e+00, -2.176024334372448e-05, -2.177928597310126e-05, 0.000000000000000e+00, -2.176024334372448e-05, -1.358429767459057e-02, 0.000000000000000e+00, -1.362427186292329e-02, -1.358062839259875e-02, 0.000000000000000e+00, -1.363054278700402e-02, -1.350457876089590e-02, 0.000000000000000e+00, -1.339749507742761e-02, -1.307597309849256e-02, 0.000000000000000e+00, -1.311571261841203e-02, -1.358107324738446e-02, 0.000000000000000e+00, -1.225512283393313e-02, -1.358107324738446e-02, 0.000000000000000e+00, -1.225512283393313e-02, -1.306004759666985e+00, 0.000000000000000e+00, -1.238199887661941e+00, -1.318687526087456e+00, 0.000000000000000e+00, -1.234025065558181e+00, -7.879187899103411e-03, 0.000000000000000e+00, -7.400749496003570e-03, -1.629499378085957e+00, 0.000000000000000e+00, -1.580463289722357e+00, -1.128654124376768e+00, 0.000000000000000e+00, -2.209801267269760e+00, -1.128654124376768e+00, 0.000000000000000e+00, -2.209801267269763e+00, -4.139672557017931e+00, 0.000000000000000e+00, -4.098382936617464e+00, -4.361979143072233e+00, 0.000000000000000e+00, -4.339951543200026e+00, -2.441109518318160e+00, 0.000000000000000e+00, -2.457196448530503e+00, -3.790757091116217e+00, 0.000000000000000e+00, -3.687829026534963e+00, -4.132406870151360e+00, 0.000000000000000e+00, -1.048348859774123e+01, -4.132406870151369e+00, 0.000000000000000e+00, -1.048348859774125e+01, -4.822708983461194e-06, 0.000000000000000e+00, -4.827136686612574e-06, -4.823924790866339e-06, 0.000000000000000e+00, -4.828314506258087e-06, -4.822736194871448e-06, 0.000000000000000e+00, -4.827148247658146e-06, -4.823780056773934e-06, 0.000000000000000e+00, -4.828218734182189e-06, -4.823364417996436e-06, 0.000000000000000e+00, -4.827734732880295e-06, -4.823364417996436e-06, 0.000000000000000e+00, -4.827734732880295e-06, -1.613132237878668e-04, 0.000000000000000e+00, -1.613410582785401e-04, -1.583584547685827e-04, 0.000000000000000e+00, -1.584903330009368e-04, -1.605644340322000e-04, 0.000000000000000e+00, -1.608628683317225e-04, -1.580796066061971e-04, 0.000000000000000e+00, -1.582876997578065e-04, -1.600283524732016e-04, 0.000000000000000e+00, -1.599161559722125e-04, -1.600283524732016e-04, 0.000000000000000e+00, -1.599161559722125e-04, -2.756111583816508e-02, 0.000000000000000e+00, -2.787029433731111e-02, -2.303512063084671e-02, 0.000000000000000e+00, -2.303140126531893e-02, -3.921686076639497e-02, 0.000000000000000e+00, -3.462236362765587e-02, -3.930161327148023e-02, 0.000000000000000e+00, -3.441056937564426e-02, -2.415167619784687e-02, 0.000000000000000e+00, -2.939574799303773e-02, -2.415167619784688e-02, 0.000000000000000e+00, -2.939574799303775e-02, -1.746164125475683e+00, 0.000000000000000e+00, -1.776960552660709e+00, -7.632217728236786e-01, 0.000000000000000e+00, -7.565740656407789e-01, -1.957570510929886e+00, 0.000000000000000e+00, -1.906862756897378e+00, -2.806047118442251e-04, 0.000000000000000e+00, -2.811335992662975e-04, -1.938104827177539e+00, 0.000000000000000e+00, -2.343006686552586e+00, -1.938104827177539e+00, 0.000000000000000e+00, -2.343006686552586e+00, -5.321266421019731e+00, 0.000000000000000e+00, -4.605744611964512e+00, -4.582273236557986e+00, 0.000000000000000e+00, -4.230700449361639e+00, -2.610237200331261e+01, 0.000000000000000e+00, -2.903821612781332e+01, -2.850786617169196e+00, 0.000000000000000e+00, -2.746442585793857e+00, -1.297955997394036e+01, 0.000000000000000e+00, -1.275082356257813e+01, -1.297955997394031e+01, 0.000000000000000e+00, -1.275082356257812e+01, -3.290469054414018e-02, 0.000000000000000e+00, -3.241957877427163e-02, -3.252052390510150e-02, 0.000000000000000e+00, -3.205145145720326e-02, -3.265422442256026e-02, 0.000000000000000e+00, -3.218047304516386e-02, -3.276794321887342e-02, 0.000000000000000e+00, -3.228774658969624e-02, -3.271120315867853e-02, 0.000000000000000e+00, -3.223412135165096e-02, -3.271120315867853e-02, 0.000000000000000e+00, -3.223412135165096e-02, -3.707550389839047e-02, 0.000000000000000e+00, -3.659230417864723e-02, -4.763804672663068e-02, 0.000000000000000e+00, -4.703524504966397e-02, -4.472436602478976e-02, 0.000000000000000e+00, -4.412864526913667e-02, -4.178914069951763e-02, 0.000000000000000e+00, -4.125652955623340e-02, -4.335636499090360e-02, 0.000000000000000e+00, -4.278162229172532e-02, -4.335636499090360e-02, 0.000000000000000e+00, -4.278162229172532e-02, -1.900062624500360e-02, 0.000000000000000e+00, -1.892558309252095e-02, -4.616241759611446e-01, 0.000000000000000e+00, -4.572365114028054e-01, -3.266375705512712e-01, 0.000000000000000e+00, -3.222471473047142e-01, -1.816004077635129e-01, 0.000000000000000e+00, -1.784148899765894e-01, -2.498457854493478e-01, 0.000000000000000e+00, -2.513867511633565e-01, -2.498457854493480e-01, 0.000000000000000e+00, -2.513867511633567e-01, -6.589431510874733e-02, 0.000000000000000e+00, -6.467745635924761e-02, -2.231483459600850e+00, 0.000000000000000e+00, -2.234354487947056e+00, -2.210639152438949e+00, 0.000000000000000e+00, -2.260560095653335e+00, -2.391322224720354e-01, 0.000000000000000e+00, -2.280262469921360e-01, -2.866828894049546e+00, 0.000000000000000e+00, -3.373277747184973e+00, -2.866828894049547e+00, 0.000000000000000e+00, -3.373277747184975e+00, -3.281675167161400e+00, 0.000000000000000e+00, -3.350339925697441e+00, -1.637134891612378e+01, 0.000000000000000e+00, -2.899954086664946e+01, -1.010456449594004e+01, 0.000000000000000e+00, -1.075709783008481e+01, -3.196198733996419e+00, 0.000000000000000e+00, -3.131129226103937e+00, -2.676107456908615e+01, 0.000000000000000e+00, -1.323648878778194e+01, -2.676107456908620e+01, 0.000000000000000e+00, -1.323648878778198e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05