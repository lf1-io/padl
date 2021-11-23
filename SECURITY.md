# Reporting vulnerabilities

Please email reports about any security related issues you find to
`padl@lf1.io`. For critical problems, you may encrypt your report (see
below).

Please use a descriptive subject line for your report email. After the initial
reply to your report, the security team will endeavor to keep you informed of
the progress being made towards a fix and announcement.

In addition, please include the following information along with your report:

* Your name and affiliation (if any).
* A description of the technical details of the vulnerabilities. It is very
  important to let us know how we can reproduce your findings.
* An explanation who can exploit this vulnerability, and what they gain when
  doing so -- write an attack scenario. This will help us evaluate your report
  quickly, especially if the issue is complex.
* Whether this vulnerability public or known to third parties. If it is, please
  provide details.

If you believe that an existing (public) issue is security-related, please send
an email to `padl@lf1.io`. The email should include the issue ID and
a short description of why it should be handled according to this security
policy.

Once an issue is reported, PADL uses the following disclosure process:

* When a report is received, we confirm the issue and determine its severity.
* If we know of specific third-party services or software based on PADL
  that require mitigation before publication, those projects will be notified.
* An advisory is prepared (but not published) which details the problem and
  steps for mitigation.
* The vulnerability is fixed and potential workarounds are identified.
* Wherever possible, the fix is also prepared for the branches corresponding to
  all releases of PADL at most one year old. We will attempt to commit
  these fixes as soon as possible, and as close together as possible.

## Encryption key for `padl@lf1.io`

If your disclosure is extremely sensitive, you may choose to encrypt your
report using the key below. Please only use this for critical security
reports.

```
-----BEGIN PGP PUBLIC KEY BLOCK-----

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
0R7t
=gNWQ
-----END PGP PUBLIC KEY BLOCK-----
```
